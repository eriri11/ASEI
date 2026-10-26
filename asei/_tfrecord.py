"""Minimal TensorFlow Example/TFRecord reader for the released MMWHS data.

The official release stores one serialized ``tf.train.Example`` per file. This
reader intentionally avoids importing TensorFlow: the project only needs the
small protobuf subset used by ``PnpAda_release_data``.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np


class TFRecordFormatError(RuntimeError):
    """Raised when a released TFRecord does not match the expected schema."""


def _read_varint(data: bytes, offset: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            raise TFRecordFormatError("protobuf varint is too long")
    raise TFRecordFormatError("truncated protobuf varint")


def _fields(data: bytes) -> Iterator[Tuple[int, int, int | bytes]]:
    """Yield protobuf ``(field_number, wire_type, value)`` tuples."""

    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        field_number, wire_type = key >> 3, key & 0x07
        if field_number <= 0:
            raise TFRecordFormatError(f"invalid protobuf field number: {field_number}")

        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise TFRecordFormatError("truncated fixed64 protobuf field")
            value = data[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise TFRecordFormatError("truncated length-delimited protobuf field")
            value = data[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise TFRecordFormatError("truncated fixed32 protobuf field")
            value = data[offset:end]
            offset = end
        else:
            raise TFRecordFormatError(f"unsupported protobuf wire type: {wire_type}")
        yield field_number, wire_type, value


def _field_value(message: bytes, number: int, wire_type: int):
    for field_number, actual_wire_type, value in _fields(message):
        if field_number == number and actual_wire_type == wire_type:
            return value
    raise TFRecordFormatError(f"missing field {number} with wire type {wire_type}")


def _feature_map(example: bytes) -> Dict[str, Tuple[str, object]]:
    """Decode the Example feature map into ``name -> (kind, value)``."""

    features = _field_value(example, 1, 2)
    result: Dict[str, Tuple[str, object]] = {}
    for field_number, wire_type, entry in _fields(features):
        if field_number != 1 or wire_type != 2:
            continue
        key = _field_value(entry, 1, 2).decode("utf-8")
        feature = _field_value(entry, 2, 2)
        feature_fields = list(_fields(feature))
        if any(number == 1 and wire == 2 for number, wire, _ in feature_fields):
            bytes_list = _field_value(feature, 1, 2)
            values = [value for number, wire, value in _fields(bytes_list) if number == 1 and wire == 2]
            result[key] = ("bytes", values)
        elif any(number == 3 and wire == 2 for number, wire, _ in feature_fields):
            int_list = _field_value(feature, 3, 2)
            values = []
            for number, wire, value in _fields(int_list):
                if number == 1 and wire == 0:
                    # int64 values in this release are non-negative dimensions
                    # except for unused bias fields, so unsigned varints are
                    # sufficient for the dimensions we consume.
                    values.append(int(value))
                elif number == 1 and wire == 2:
                    packed = value
                    offset = 0
                    while offset < len(packed):
                        item, offset = _read_varint(packed, offset)
                        values.append(int(item))
            result[key] = ("int64", values)
        else:
            raise TFRecordFormatError(f"unsupported Feature kind for key: {key}")
    return result


def iter_tfrecord_examples(path: str | Path) -> Iterator[bytes]:
    """Yield serialized Example payloads from a TFRecord file."""

    with Path(path).open("rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise TFRecordFormatError(f"truncated length header: {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            length_crc = handle.read(4)
            payload = handle.read(length)
            payload_crc = handle.read(4)
            if len(length_crc) != 4 or len(payload) != length or len(payload_crc) != 4:
                raise TFRecordFormatError(f"truncated TFRecord payload: {path}")
            yield payload


def decode_mmwhs_example(example: bytes) -> Dict[str, np.ndarray | Dict[str, int]]:
    """Decode one released MMWHS example into channel-first arrays."""

    feature_map = _feature_map(example)

    def bytes_feature(name: str) -> bytes:
        kind, values = feature_map[name]
        if kind != "bytes" or len(values) != 1:
            raise TFRecordFormatError(f"Expected one bytes value for {name}")
        return values[0]

    def int_feature(name: str) -> int:
        kind, values = feature_map[name]
        if kind != "int64" or len(values) != 1:
            raise TFRecordFormatError(f"Expected one integer value for {name}")
        return int(values[0])

    image_shape = tuple(int_feature(f"dsize_dim{i}") for i in range(3))
    label_shape = tuple(int_feature(f"lsize_dim{i}") for i in range(3))
    if image_shape != label_shape:
        raise TFRecordFormatError(f"Image/label shape mismatch: {image_shape} vs {label_shape}")

    image_raw = bytes_feature("data_vol")
    label_raw = bytes_feature("label_vol")
    expected_bytes = int(np.prod(image_shape)) * np.dtype("<f4").itemsize
    if len(image_raw) != expected_bytes or len(label_raw) != expected_bytes:
        raise TFRecordFormatError(
            f"Unexpected raw size for shape {image_shape}: image={len(image_raw)}, label={len(label_raw)}"
        )

    image = np.frombuffer(image_raw, dtype="<f4").reshape(image_shape).copy()
    label_float = np.frombuffer(label_raw, dtype="<f4").reshape(label_shape)
    if not np.allclose(label_float, np.rint(label_float)):
        raise TFRecordFormatError("Labels contain non-integer values")
    label = np.rint(label_float).astype(np.int64, copy=True)

    # Convert released HWC arrays to channel-first samples.
    image = np.moveaxis(image, -1, 0)
    label = np.moveaxis(label, -1, 0)
    return {
        "image": image,
        "label": label,
        "height": image_shape[0],
        "width": image_shape[1],
        "num_slices": image_shape[2],
    }


def read_single_mmwhs_tfrecord(path: str | Path) -> Dict[str, np.ndarray | Dict[str, int]]:
    examples = list(iter_tfrecord_examples(path))
    if len(examples) != 1:
        raise TFRecordFormatError(f"Expected one Example in {path}, found {len(examples)}")
    return decode_mmwhs_example(examples[0])
