import csv
import json

from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', default=None,
                    help='Path to input file')
flags.DEFINE_string('output_path', default=None,
                    help='Path to save output file')


def main(argv):
    input_file = open(FLAGS.input_path, 'r')
    output_file = open(FLAGS.output_path, 'w')

    csv_reader = csv.DictReader(input_file,
                                delimiter='\t')

    for data in tqdm(csv_reader, desc='Preprocessing'):
        processed_data = data.copy()
        processed_data['document'] = data['document']
        processed_data['label'] = data['label']
        del processed_data['id']

        output_file.write(json.dumps(processed_data, ensure_ascii=False))
        output_file.write('\n')

    input_file.close()
    output_file.close()


if __name__ == '__main__':
    flags.mark_flags_as_required(['input_path', 'output_path'])
    app.run(main)
