#!/usr/bin/env python3
"""Per-tensor quant-type histogram straight from the GGUF header.

The throughput gap between the two revisions is not explained by file size or by
drafter acceptance, so the remaining candidate is which block types each revision
picked per tensor - some are on much faster SYCL paths than others.
"""
import struct, sys, collections
T={0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
   10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
   16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",
   22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",29:"IQ1_M",
   30:"BF16",34:"TQ1_0",35:"TQ2_0"}
def rd(f,fmt): n=struct.calcsize(fmt); return struct.unpack(fmt,f.read(n))
def rstr(f): (l,)=rd(f,"<Q"); return f.read(l).decode("utf-8",errors="replace")
def skipval(f,t):
    S={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t==8: rstr(f); return
    if t==9:
        (et,)=rd(f,"<I"); (n,)=rd(f,"<Q")
        for _ in range(n): skipval(f,et)
        return
    f.read(S[t])
p=sys.argv[1]
with open(p,"rb") as f:
    magic=f.read(4); assert magic==b"GGUF", magic
    ver,ntensor,nkv = rd(f,"<IQQ")
    for _ in range(nkv):
        rstr(f); (vt,)=rd(f,"<I"); skipval(f,vt)
    hist=collections.Counter(); bytes_by=collections.Counter()
    for _ in range(ntensor):
        name=rstr(f); (nd,)=rd(f,"<I")
        dims=rd(f,"<"+"Q"*nd); (tt,)=rd(f,"<I"); rd(f,"<Q")
        n=1
        for d in dims: n*=d
        hist[T.get(tt,str(tt))]+=1; bytes_by[T.get(tt,str(tt))]+=n
print(p.split("/")[-2]+"/"+p.split("/")[-1], "tensors:", ntensor)
for k,v in sorted(hist.items(), key=lambda x:-bytes_by[x[0]]):
    print(f"   {k:9s} {v:4d} tensors  {bytes_by[k]/1e9:7.2f} G-elems")
