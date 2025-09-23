import csv

# Input and output file paths
input_file = "./data/values_emotions_dataset_joy_modified_modified.csv"
output_file = "./data/values_emotions_dataset_joy_modified_modified_reversed.csv"

with (
    open(input_file, "r", newline="", encoding="utf-8") as infile,
    open(output_file, "w", newline="", encoding="utf-8") as outfile,
):

    reader = csv.reader(infile, delimiter="|")
    writer = csv.writer(outfile, delimiter="|")

    for row in reader:
        # Swap first and third column
        if len(row) >= 3:
            row[0], row[2] = row[2], row[0]
        writer.writerow(row)

print(f"Reordered CSV saved to {output_file}")
