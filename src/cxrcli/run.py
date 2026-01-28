import argparse
from pathlib import Path
from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CXR Multimodal Baseline CLI")
    parser.add_argument("--csv", type=Path, required=True, help="Path to CSV metadata file")
    parser.add_argument("--imgdir", type=Path, required=True, help="Directory containing CXR images")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--output", type=Path, default="outputs/", help="Directory to save outputs")

    args = parser.parse_args()

    # Run the main pipeline with provided arguments
    main(csv_path=args.csv, image_dir=args.imgdir, batch_size=args.batch_size, output_dir=args.output)