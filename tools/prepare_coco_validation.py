import urllib.request,zipfile,json,hashlib,concurrent.futures,io,argparse
from pathlib import Path
parser=argparse.ArgumentParser(description='Prepare the frozen 512-image COCO val2017 subset without generated labels.')
parser.add_argument('--output',type=Path,required=True)
root=parser.parse_args().output;root.mkdir(parents=True,exist_ok=True)
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
archive=root/'annotations_trainval2017.zip'
url='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip'
class RemoteZip(io.RawIOBase):
 def __init__(self):
  self.position=0
  with op.open(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=30) as r:
   assert r.status==206
   self.length=int(r.headers['Content-Range'].split('/')[-1])
 def seekable(self):return True
 def tell(self):return self.position
 def seek(self,offset,whence=0):
  self.position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
  return self.position
 def read(self,size=-1):
  end=self.length-1 if size<0 else min(self.position+size,self.length)-1
  if end<self.position:return b''
  with op.open(urllib.request.Request(url,headers={'Range':f'bytes={self.position}-{end}'}),timeout=60) as r:
   assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {self.position}-{end}/')
   b=r.read();assert len(b)==end-self.position+1
  self.position+=len(b);return b
annotation_path=root/'instances_val2017.json'
if annotation_path.exists():raw=annotation_path.read_bytes()
else:
 with zipfile.ZipFile(RemoteZip()) as z:raw=z.read('annotations/instances_val2017.json')
 print('Verified annotation ZIP entry',len(raw),flush=True)
(root/'instances_val2017.json').write_bytes(raw)
if hashlib.sha256(raw).hexdigest()!='e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f':
 raise ValueError('Annotation file differs from the frozen official COCO val2017 source')
data=json.loads(raw); categories={c['id']:c['name'] for c in data['categories']}
truth={i['id']:set() for i in data['images']}
for a in data['annotations']:truth[a['image_id']].add(categories[a['category_id']])
key=lambda i:hashlib.sha256(('ram-clip-validation-v1:'+str(i['id'])).encode()).hexdigest()
selected=sorted(data['images'],key=key)[:512]
split={i['id']:('development' if j<128 else 'evaluation') for j,i in enumerate(selected)}
images=root/'images';images.mkdir(exist_ok=True)
def fetch(i):
 p=images/i['file_name'];u='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+i['file_name']
 if not p.exists():
  with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(u,timeout=60) as r:p.write_bytes(r.read())
 from PIL import Image
 with Image.open(p) as im:
  im.load()
  assert im.size==(i['width'],i['height'])
 return {'image':p.relative_to(root).as_posix(),'image_id':i['id'],'split':split[i['id']],
         'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'truth':sorted(truth[i['id']]),'source_url':u}
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for j,r in enumerate(pool.map(fetch,selected)):
  rows.append(r)
  if (j+1)%32==0:print('Images verified',j+1,flush=True)
result={'source':url,'annotation_sha256':hashlib.sha256(raw).hexdigest(),
 'selection':'512 smallest SHA256(ram-clip-validation-v1:<image_id>); first128 development, remaining384 evaluation',
 'images':rows}
(root/'manifest.json').write_text(json.dumps(result,indent=2))
print('Ready',len(rows),flush=True)
