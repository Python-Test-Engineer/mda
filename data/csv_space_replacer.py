import csv
import sys


def replace_spaces_in_rule_field(input_file, output_file=None):
    """
    Read a pipe-delimited CSV and replace spaces with underscores in the second field (Rule column) only.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional, defaults to input_file with _modified suffix)
    """

    # Generate output filename if not provided
    if output_file is None:
        if input_file.endswith(".csv"):
            output_file = input_file.replace(".csv", "_modified.csv")
        else:
            output_file = input_file + "_modified"

    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            with open(output_file, "w", encoding="utf-8", newline="") as outfile:
                # Use pipe delimiter for reading and writing
                reader = csv.reader(infile, delimiter="|")
                writer = csv.writer(outfile, delimiter="|")

                row_count = 0
                modified_count = 0

                for row in reader:
                    if len(row) >= 2:  # Ensure row has at least 2 columns
                        original_rule = row[1]
                        # Replace spaces with underscores in the second field only
                        row[1] = row[1].replace(" ", "_")

                        # Count modifications for reporting
                        if original_rule != row[1]:
                            modified_count += 1

                    writer.writerow(row)
                    row_count += 1

                print(f"Processing complete!")
                print(f"Total rows processed: {row_count}")
                print(f"Rules modified: {modified_count}")
                print(f"Output saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False

    return True


def main():
    """Main function to handle command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file> [output_file]")
        print("Example: python script.py data.csv")
        print("Example: python script.py data.csv modified_data.csv")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    replace_spaces_in_rule_field(input_file, output_file)


if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) < 2:
        # Demo with hardcoded filename - change this to your actual CSV filename
        input_filename = (
            "values_emotions_dataset_joy.csv"  # Change this to your CSV filename
        )
        print(f"Demo mode: Processing '{input_filename}'")
        replace_spaces_in_rule_field(input_filename)
    else:
        main()
