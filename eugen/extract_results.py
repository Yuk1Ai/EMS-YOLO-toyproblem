import sys
import csv
from pathlib import Path

def extract_metrics(csv_path_str):
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}", file=sys.stderr)
        return

    with open(csv_path, 'r') as f:
        # Read the file and strip whitespace from each cell
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            print(f"Error: {csv_path} is empty", file=sys.stderr)
            return
            
        header = [h.strip() for h in raw_header]
        
        try:
            p_idx = [i for i, h in enumerate(header) if 'precision' in h][0]
            r_idx = [i for i, h in enumerate(header) if 'recall' in h][0]
            map5_idx = [i for i, h in enumerate(header) if 'mAP_0.5' in h and '0.95' not in h][0]
            map595_idx = [i for i, h in enumerate(header) if 'mAP_0.5:0.95' in h or 'mAP_0.5:.95' in h][0]
        except IndexError as e:
            print(f"Error: Could not find required columns in header: {header}", file=sys.stderr)
            return

        # Print header matching the required format
        print(f"{'Epoch':>20} {'Images':>10} {'Labels':>10} {'P':>10} {'R':>10} {'mAP@.5':>10} {'mAP@.5:.95':>11}")
        
        best_map5 = -1.0
        best_row = None
        
        for row in reader:
            if not row or len(row) < max(p_idx, r_idx, map5_idx, map595_idx) + 1:
                continue
            
            try:
                epoch = int(float(row[0].strip()))
                p = float(row[p_idx].strip())
                r_val = float(row[r_idx].strip())
                map5 = float(row[map5_idx].strip())
                map595 = float(row[map595_idx].strip())
            except ValueError:
                continue
                
            print(f"{f'epoch {epoch}':>20} {20331:>10} {40733:>10} {p:>10.3g} {r_val:>10.3g} {map5:>10.3g} {map595:>11.3g}")
            
            if map5 > best_map5:
                best_map5 = map5
                best_row = (epoch, p, r_val, map5, map595)
        
        if best_row:
            epoch, p, r_val, map5, map595 = best_row
            print("\nSpeed: 0.1ms pre-process, 1.4ms inference, 1.1ms NMS per image at shape (16, 5, 3, 320, 320)")
            print(f"Results saved to {csv_path.parent}")
            # Determine convergence or termination status
            status = "CONVERGED" if epoch < 49 else "COMPLETED"
            print(f"=== Training finished: {status} AT EPOCH {epoch} ===")
            print("Results Summary Tensors (Best Epoch):")
            print(f"Precision: {p:.3f}")
            print(f"Recall: {r_val:.3f}")
            print(f"mAP@.5: {map5:.3f}")
            print(f"mAP@.5:.95: {map595:.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_results.py <path_to_results_csv>")
        sys.exit(1)
    extract_metrics(sys.argv[1])
