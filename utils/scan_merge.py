from pathlib import Path
import shutil


def merge2seq(exp_dir, val_mode):
    seq_dict = dict()
    result_dir = Path(exp_dir, 'results', val_mode)
    
    single_results = list(result_dir.iterdir())
    for sr in single_results:
        if sr.is_dir():
            #pred = DSeq.load_from_file(sr, 'pred_' + sr.stem)
            gt = DSeq.load_from_file(sr,  sr.stem, scan_fields=[], object_fields=[])
        
            if gt.seq_name not in seq_dict:
                seq_dict[gt.seq_name] = dict()
            for scan_idx in gt.scans:
                seq_dict[gt.seq_name][scan_idx] = sr

    for seq_name in seq_dict:
        sscan_inds = sorted(list(seq_dict[seq_name].keys()))
        res_path = seq_dict[seq_name][sscan_inds[0]]
        gt_seq = DSeq.load_from_file(res_path,  res_path.stem)
        pred_seq = DSeq.load_from_file(res_path, 'pred_' + res_path.stem)
        for scan_idx in sscan_inds[1:]:
            res_path = seq_dict[seq_name][scan_idx]
            gt_seq.scans[scan_idx] = DSeq.load_from_file(res_path, res_path.stem).scans[scan_idx]
            pred_seq.scans[scan_idx] = DSeq.load_from_file(res_path, 'pred_' + res_path.stem).scans[scan_idx]

        gt_seq.serialize(Path(result_dir,seq_name),seq_name)
        pred_seq.serialize(Path(result_dir,seq_name),'pred_'+seq_name)

    for single_result in single_results:
        shutil.rmtree(str(single_result))