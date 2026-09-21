"""Materialize only official TRAIN rows for selected regression tasks.

No test/validation targets are loaded. ARFF rows outside selected TRAIN IDs
are skipped before tokenization. Data are written into a new per-run directory.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from train_split_audit import _load_openml_split_columns, _integer_split_column, _arff_attribute_name, _split_arff_row

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def safe_array(path):
    # These are established, trusted benchmark files, not supplied model pickles.
    try:return np.load(path,allow_pickle=False)
    except ValueError as e:
        if 'Object arrays cannot be loaded' not in str(e):raise
        x=np.load(path,allow_pickle=True)
        if not all(v is None or isinstance(v,(str,int,float,np.generic)) for v in x.flat):
            raise ValueError(f'non-scalar benchmark objects: {path}')
        return x

def arff_train(src,expected_rows):
    membership=json.loads((src/'membership.json').read_text())
    target=membership.get('target_feature')
    if not target:raise ValueError(f'missing target: {src}')
    splits=_load_openml_split_columns(src/'splits.arff')
    repeat=_integer_split_column(splits['repeat'],'repeat')
    fold=_integer_split_column(splits['fold'],'fold')
    rid=_integer_split_column(splits['rowid'],'rowid')
    r=int(repeat.min()); f=int(fold[repeat==r].min())
    selected=(repeat==r)&(fold==f)
    train_ids=rid[selected & np.array([t.upper()=='TRAIN' for t in splits['type']])]
    test_ids=rid[selected & np.array([t.upper()=='TEST' for t in splits['type']])]
    assert len(train_ids)==expected_rows and len(set(train_ids))==len(train_ids)
    assert not set(train_ids)&set(test_ids)
    attrs=[]; numeric=[]; rows={}; body=False; rownum=0; wanted=set(train_ids)
    with (src/'data.arff').open() as handle:
        for raw in handle:
            line=raw.strip()
            if not line or line.startswith('%'):continue
            if not body:
                if line.lower().startswith('@attribute'):
                    attrs.append(_arff_attribute_name(raw))
                    numeric.append(bool(re.search(r'\s(?:numeric|real|integer)\s*$',line,re.I)))
                elif line.lower()=='@data':body=True
                continue
            if rownum in wanted:
                assert not line.startswith('{'), 'sparse ARFF requires explicit adapter'
                values=_split_arff_row(raw)
                assert len(values)==len(attrs)
                rows[rownum]=values
            rownum+=1
    assert set(rows)==wanted
    target_idx=[i for i,n in enumerate(attrs) if n.casefold()==str(target).casefold()]
    assert len(target_idx)==1
    ti=target_idx[0]
    arr=np.array([rows[int(i)] for i in train_ids],dtype=str)
    y=pd.to_numeric(pd.Series(arr[:,ti]),errors='raise').to_numpy(dtype=float)
    frame={}; cats=[]
    for i,name in enumerate(attrs):
        if i==ti:continue
        column=f'feature_{i}'
        vals=pd.Series(arr[:,i]).replace('?',np.nan)
        if numeric[i]:frame[column]=pd.to_numeric(vals,errors='raise')
        else:frame[column]=vals; cats.append(column)
    return pd.DataFrame(frame),y,cats,{'repeat':r,'fold':f,'train_row_ids_sha256':hashlib.sha256(train_ids.tobytes()).hexdigest(), 'test_row_overlap':0, 'membership':membership},[src/'data.arff',src/'splits.arff',src/'membership.json']

def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists(), f'output already exists: {a.output}'
    a.output.mkdir(parents=True)
    selection=json.loads(a.selection.read_text()); entries=[]
    for d in selection['datasets']:
        suite='talent' if d['suite']=='TALENT' else d['suite']
        parent=a.root/suite/'regression'
        matches=[p for p in parent.iterdir() if p.name.strip()==d['dataset'].strip() and p.is_dir()]
        assert len(matches)==1,(d['dataset'],[str(p) for p in matches])
        src=matches[0]; provenance={};files=[]
        if suite=='talent':
            y=safe_array(src/'y_train.npy').reshape(-1).astype(float);frame={};cats=[];files=[src/'y_train.npy']
            for prefix in ['N','C']:
                path=src/f'{prefix}_train.npy'
                if not path.is_file():continue
                arr=safe_array(path)
                if arr.ndim==1:arr=arr[:,None]
                assert len(arr)==len(y)
                files.append(path)
                for j in range(arr.shape[1]):
                    key=f'{prefix}{j}';frame[key]=arr[:,j]
                    if prefix=='C':cats.append(key)
            frame=pd.DataFrame(frame)
            provenance={'split':'original TALENT train only; val/test unopened'}
        elif suite=='BCCO':
            paths=sorted(src.glob('*_train.csv'));assert len(paths)==1
            frame=pd.read_csv(paths[0]);cols=[c for c in frame if c.lower()=='target'];assert len(cols)==1
            y=pd.to_numeric(frame.pop(cols[0]),errors='raise').to_numpy(dtype=float)
            cats=[c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])];files=paths
            provenance={'split':'original BCCO train only; test unopened'}
        else:
            frame,y,cats,provenance,files=arff_train(src,int(d['n_train']))
        assert len(y)==int(d['n_train']),(d['dataset'],len(y),d['n_train'])
        assert np.isfinite(y).all() and np.std(y)>0 and frame.shape[1]>0
        ident=re.sub(r'[^A-Za-z0-9_.-]+','_',suite+'__'+d['dataset'])
        path=a.output/(ident+'_train.csv');assert '__target__' not in frame
        frame['__target__']=y;frame.to_csv(path,index=False)
        entry={'id':ident,'split':'train','train_csv':str(path.resolve()),'target_column':'__target__','categorical_columns':cats,'rows':len(y),'features':len(frame.columns)-1,'suite':suite,'dataset':d['dataset'],'sha256':digest(path),'source_files':[{'path':str(f),'sha256':digest(f)} for f in files], 'provenance':provenance}
        entries.append(entry);print(json.dumps({'prepared':ident,'rows':len(y),'features':entry['features']}),flush=True)
    result={'datasets':entries,'train_only':True,'test_labels_used':False,'selection':selection,'format_version':1}
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'prepared_count':len(entries),'manifest':str(a.output/'manifest.json')}))
if __name__=='__main__':main()
