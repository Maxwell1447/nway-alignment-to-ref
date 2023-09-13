import torch
import libnway
import libnway.nway


def nway_align(src, tms, bos=0, eos=2, unk=3, pad=1, k=10, max_valency=10):
    voc_to_id = dict()
    for w in src + [e for tm in tms for e in tm]:
        if not w in voc_to_id:
            voc_to_id[w] = 4 + len(voc_to_id)

    # print(voc_to_id)

    max_len = max(len(src), *[len(tm) for tm in tms]) + 2
    # print(max_len)

    src_i = torch.tensor([bos] + [voc_to_id[w] for w in src] + [eos])
    tms_i = [torch.tensor([bos] + [voc_to_id[w] for w in tm] + [eos]) for tm in tms]

    src_t = torch.full((1, max_len), pad, dtype=torch.long)
    tms_t = torch.full((1, len(tms), max_len), pad, dtype=torch.long)

    src_t[0, :len(src_i)] = src_i

    for n in range(len(tms_i)):
        tms_t[0, n, :len(tms_i[n])] = tms_i[n]
    
    # print(src_t)
    # print(tms_t)

    ops = libnway.nway.MultiLevEditOps(tms_t.cpu(), src_t.cpu(), k, max_valency, pad, unk)
    masked_src_cov = ops.get_s_cmb()

    # print(masked_src_cov)

    # print(1 - (masked_src_cov == 3).sum().item() / len(src))

    return 1 - (masked_src_cov == 3).sum().item() / len(src)
