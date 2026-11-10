# Resolve a "HEAD = Q5_K comment / ours = whole-tensor <T>_moe fn" conflict by porting ours to 32 MiB chunks.
import re,sys
p='ggml/src/ggml-sycl/ggml-sycl.cpp'
s=open(p).read()
m=re.search(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n',s,flags=re.S)
head,ours=m.group(1),m.group(2)
# split ours: function + trailing Q5_K comment
fn_end=ours.rindex('\n}\n')+3
fn=ours[:fn_end]
fn=re.sub(r'(// Reorder each \S+ expert slice into [^\n.]*)\.\n', r'\1, a few experts at a time (see Q4_K above).\n', fn, count=1)
fn=re.sub(r'    const int    blocks_per_expert = (.*?);\n    const size_t total_bytes       = expert_bytes \* \(size_t\) n_expert;\n',
 r'''    const int     blocks_per_expert = \1;
    const size_t  max_chunk_bytes   = 32u << 20;
    const int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));
''',fn)
fn=fn.replace('''    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\\n", __func__, total_bytes);''',
'''    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    if (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\\n", __func__,
                      (size_t) chunk_experts * expert_bytes);''')
a=fn.index('    sycl::event copy_event;')
b=fn.index('    return true;\n}')
body=fn[a:b]
body=body.replace('memcpy(tmp_buf, data_device, total_bytes)','memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)')
body=body.replace('blocks_per_expert * (int) n_expert','blocks_per_expert * (int) n_chunk')
body=body.replace('data_device + (size_t) e * expert_bytes','chunk + (size_t) e * expert_bytes')
body='\n'.join(('    '+l if l.strip() else l) for l in body.split('\n'))
loop='''    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    for (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        const int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

'''+body.rstrip()+'\n    }\n'
fn=fn[:a]+loop+fn[b:]
for bad in ('total_bytes','data_device + (size_t) e '):
    assert bad not in fn, bad
s=s[:m.start()]+fn+'\n'+head+s[m.end():]
open(p,'w').write(s)
print(fn)
