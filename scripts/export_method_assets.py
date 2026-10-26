"""Export reproducible, GT-free GLFP/ECSS figure assets from a canonical cache."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asei.config import load_config
from asei.data import normalize
from asei.adaptation import TargetAugmentation
from asei.ecss import construct_sets
from asei.medsam import EmbeddingCache
from asei.models import build_model
from asei.utils import load_checkpoint,load_npz,sha_file,seed_everything,write_json,unpack

NAMES = ['MYO', 'LA', 'LV', 'AA']
COLORS = {0: (0, 0, 0), 1: (242, 91, 101), 2: (87, 196, 119),
          3: (84, 191, 225), 4: (243, 206, 89), 255: (65, 65, 65)}
SET_COLOR = (173, 133, 225)


def colored(label):
    rgb = np.zeros((*label.shape, 3), np.uint8)
    for c, color in COLORS.items():
        rgb[label == c] = color
    return rgb


def overlay(image, label):
    rgb = np.repeat(image[..., None], 3, axis=2).astype(float)
    for c in range(1, 5):
        m = label == c
        rgb[m] = .38 * rgb[m] + .62 * np.array(COLORS[c])
    return np.round(rgb).astype(np.uint8)


def fragmentation(label):
    result = {}
    for c, name in enumerate(NAMES, 1):
        components, count = ndimage.label(label == c)
        areas = np.bincount(components.ravel())[1:]
        result[name] = dict(components=int(count), pixels=int(areas.sum()),
                            component_areas=sorted(areas.tolist(), reverse=True))
    return result


def load_model(path, config, device):
    model = build_model(config).to(device)
    load_checkpoint(path, model)
    return model.eval().requires_grad_(False)


def prompts(gray, groups):
    image = Image.fromarray(gray).convert('RGB').resize((1024, 1024), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(image)
    for boxes, classes, dashed in groups:
        for box, c in zip(boxes, classes):
            x0, y0, x1, y1 = [float(v) * 4 for v in box]
            color = COLORS[int(c) + 1]
            if not dashed:
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=4)
            else:
                for a, b, horizontal in [(x0, x1, True), (y0, y1, False)]:
                    for start in np.arange(a, b, 12):
                        stop = min(start + 7, b - 1)
                        for fixed in ([y0, y1 - 1] if horizontal else [x0, x1 - 1]):
                            segment = (start, fixed, stop, fixed) if horizontal else (fixed, start, fixed, stop)
                            draw.line(segment, fill=color, width=4)
    return image


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('config','sample','labels','round1','responses','teacher','student','output'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--verify-set-bank',type=Path)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(4);seed_everything(42)
    config=load_config(args.config)
    with np.load(args.sample,allow_pickle=False) as archive:
        image=normalize(archive['image'],(-2,-1))
    if image.ndim==2:image=image[None]
    center=0 if image.shape[0]==1 else config['data'].get('center_slice',1)
    tensor=torch.from_numpy(image[center:center+1].copy())
    raw=load_npz(args.responses);final=load_npz(args.labels);first=load_npz(args.round1)
    assert EmbeddingCache.image_key(tensor.numpy())==str(raw['image_sha256'])
    settings=config['adaptation']['ecss']
    result=construct_sets(final['pseudo'],final['weight'],raw,settings,glfp_options=config['adaptation']['glfp'])
    if args.verify_set_bank:
        saved=load_npz(args.verify_set_bank)
        for key in ('set_bits','set_weight','original_foreground_mass'):
            assert np.array_equal(saved[key],result[key]),key
    output=args.output;data=output/'_data';data.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(36)
    views=TargetAugmentation(False)({'image':tensor})
    probabilities={};prediction={}
    checkpoints=dict(teacher=args.teacher,student=args.student,frozen=Path(config['source_checkpoint']))
    for role,path in checkpoints.items():
        model=load_model(path,config,torch.device(args.device))
        view=views['strong_image'] if role=='student' else tensor
        probabilities[role]=model(view[None].to(args.device)).softmax(1)[0].cpu().numpy()
        prediction[role]=probabilities[role].argmax(0).astype(np.uint8)
        del model
    p,w=final['pseudo'],final['weight'];hard=(p!=255)&(w>0)
    sets=result['set_weight']>0;remaining=~hard&~sets
    assert not (hard&sets).any() and (hard|sets|remaining).all()
    gray=np.round(tensor[0].numpy().clip(0,1)*255).astype(np.uint8)
    entries=[]

    def save(name,title,pixels,categorical=True):
        im=pixels if isinstance(pixels,Image.Image) else Image.fromarray(pixels)
        if im.size==(256,256):
            im.save(data/(name+'_256.png'))
            im=im.resize((1024,1024),Image.Resampling.NEAREST if categorical else Image.Resampling.BICUBIC)
        im.save(output/(name+'.png'));entries.append((name,title))

    save('01_target_image','Target image',gray,False)
    for num,role,key in ((2,'Weak','weak_image'),(3,'Strong','strong_image')):
        save(f'{num:02d}_{role.lower()}_view',role+' view',np.round(views[key][0].numpy().clip(0,1)*255).astype(np.uint8),False)
    save('04_global_prompts','Global prompts',prompts(gray,[(raw['global_boxes'],raw['global_classes'],False)]))
    for num,title,prefix,dashed in ((5,'Local round 1','local',False),(6,'Feedback rounds','extra',True)):
        save('05_local_round1' if num==5 else '06_local_feedback',title,prompts(gray,[(raw[prefix+'_boxes'],raw[prefix+'_classes'],dashed)]))
    save('07_local_combined','All local rounds',prompts(gray,[(raw['local_boxes'],raw['local_classes'],False),(raw['extra_boxes'],raw['extra_classes'],True)]))
    for num,role in ((8,'teacher'),(9,'student')):
        save(f'{num:02d}_{role}_prediction',role.title()+' output',colored(prediction[role]))
    save('10_pseudo_round1','GLFP round 1',colored(first['pseudo']))
    save('11_pseudo_final','GLFP final hard labels',colored(p))
    save('12_pseudo_overlay','Final hard foreground',overlay(gray,p))
    save('13_frozen_prediction','Frozen prompt model',colored(prediction['frozen']))
    cores=unpack(raw['cores'],(4,*p.shape));anchors=unpack(raw['extra_anchors'],(4,*p.shape))
    for num,key,masks in ((14,'stable_cores',cores),(16,'feedback_anchors',anchors)):
        label=np.zeros_like(p)
        for c in range(4):label[masks[c]]=c+1
        save(f'{num:02d}_{key}',key.replace('_',' ').title(),colored(label))
        if num==14:save('15_stable_cores_overlay','Stable cores overlay',overlay(gray,label))
    combined=colored(p);combined[sets]=SET_COLOR
    save('17_ecss_regions','Hard labels + ECSS sets',combined)
    palette={0:(35,40,46),**{c:COLORS[c] for c in range(1,5)}}
    hard_rgb=np.zeros((*p.shape,3),np.uint8)
    for c,color in palette.items():hard_rgb[hard&(p==c)]=color
    for key,mask,color,title in (('set_region',sets,SET_COLOR,'ECSS set region C'),
        ('ignore_region',remaining,(144,177,204),'Remaining unknown M'),
        ('geometry_12px',result['geometric'],(242,184,75),'12px geometric region'),
        ('conflict_excluded',result['conflict']&result['unknown'],(255,109,0),'Excluded conflicts')):
        rgb=np.zeros_like(hard_rgb);rgb[mask]=color;save(key,title,rgb)
    save('hard_region','Original hard region H',hard_rgb)
    evidence=output/'evidence';evidence.mkdir(exist_ok=True)
    for c,name in enumerate(NAMES):
        for key in ('global_positive','global_negative','local_positive','local_negative'):
            rgb=np.full((*p.shape,3),245,np.uint8)
            if key.startswith('local'):rgb[~result['local_observed'][c]]=(220,225,230)
            elif not result['global_observed'][c]:rgb[:]=(220,225,230)
            rgb[result[key][c]]=(75,158,133) if key.endswith('positive') else (199,125,133)
            Image.fromarray(rgb).resize((1024,1024),Image.Resampling.NEAREST).save(evidence/f'{name}_{key}.png')
    probability_dir=output/'probability_channels';probability_dir.mkdir(exist_ok=True)
    for role in ('teacher','student'):
        stack=Image.new('RGBA',(1024,1024),(0,0,0,0))
        for j,c in enumerate((0,2,4,3,1)):
            rgba=np.round(colormaps['magma'](probabilities[role][c])*255).astype(np.uint8)
            channel=Image.fromarray(rgba)
            channel.save(probability_dir/f'{role}_class{c}.png')
            channel=channel.resize((820,820),Image.Resampling.BILINEAR)
            draw=ImageDraw.Draw(channel);draw.rectangle((0,0,819,819),outline=(110,120,130,255),width=2)
            stack.alpha_composite(channel,(20+40*j,20+40*j))
        stack.save(output/f'{role}_probability_stack.png')
    np.savez_compressed(output/'soft_probabilities.npz',teacher=probabilities['teacher'],student=probabilities['student'])
    np.savez_compressed(output/'region_partition.npz',H=hard,C=sets,M=remaining,pseudo=p,weight=w,set_bits=result['set_bits'],set_weight=result['set_weight'])
    np.savez_compressed(data/'labels_and_views.npz',original=tensor.numpy(),weak=views['weak_image'].numpy(),strong=views['strong_image'].numpy(),
        teacher=prediction['teacher'],student=prediction['student'],frozen=prediction['frozen'],pseudo=p,pseudo_round1=first['pseudo'],weight=w,
        stable_cores=cores,feedback_anchors=anchors,**result)
    for source,name in ((args.responses,'responses.npz'),(args.labels,'hard_targets.npz'),(args.round1,'round1.npz')):
        import shutil
        shutil.copy2(source,data/name)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    tile,header,gap=256,42,14;cols=4;rows=(len(entries)+cols-1)//cols
    canvas=Image.new('RGB',(cols*(tile+gap)+gap,rows*(tile+header+gap)+gap),'white');draw=ImageDraw.Draw(canvas)
    for i,(name,title) in enumerate(entries):
        x=gap+i%cols*(tile+gap);y=gap+i//cols*(tile+header+gap)
        draw.text((x,y+8),title,font=font,fill='black')
        with Image.open(output/(name+'.png')) as im:canvas.paste(im.convert('RGB').resize((tile,tile),Image.Resampling.NEAREST),(x,y+header))
    canvas.save(output/'overview.png')
    source_paths={k:str(v.resolve()) for k,v in vars(args).items() if isinstance(v,Path)}
    write_json(output/'metadata.json',dict(framework='ASEI',modules=['GLFP','ECSS'],sample=str(args.sample),
        image_sha256=str(raw['image_sha256']),options=settings,augmentation_seed=36,
        checkpoint_SHA256={k:dict(path=str(v),sha256=sha_file(v)) for k,v in checkpoints.items()},
        source_paths=source_paths,source_SHA256={k:sha_file(v) for k,v in vars(args).items() if isinstance(v,Path) and v.is_file()},
        GT_loaded=False,original_hard_targets_unchanged=True,extra_MedSAM_calls=0,
        postprocessing='none; native training masks, no LCC or morphology',
        sample_selection='Previously selected illustration slice; not a representative/random sample or quantitative comparison.',
        set_pixels=int(sets.sum()),hard_pixels=int(hard.sum()),remaining_unknown_pixels=int(remaining.sum()),
        view_coordinates='Canonical input coordinates; before optional CT spatial warping',
        evidence_semantics='Exactly the same four accepted GLFP evidence masks; all local rounds accumulated per query.'))
    (output/'README.md').write_text('# ASEI = GLFP + ECSS 素材\n\n'
        '单图为1024×1024 PNG；teacher/student概率堆叠为透明底，均来自本次ECSS权重。'
        'hard_region、set_region、ignore_region对应互斥的H/C/M；紫色为集合监督，不是新器官。'
        'evidence目录为四器官的GLFP与ECSS共享的四类严格正/负证据，局部框外和未查询全局类为灰色未观测。'
        '原始数组、输入SHA及检查点来源见_data、metadata.json。本脚本不读取GT，不重新调用MedSAM。'
        '这是之前选定病例的流程素材，不用于证明平均性能；分割掩码未做LCC或形态修饰。\n')
    print(json.dumps(dict(output=str(output),set_pixels=int(sets.sum()),verified_training_bank=args.verify_set_bank is not None)),flush=True)


if __name__=='__main__':main()
