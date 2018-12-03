import os
from optparse import OptionParser
import ffmpy
import glob

def split_by_seconds(input_folder, output_folder, split_length, vcodec="copy", **kwargs):

    input_files = glob.glob(input_folder + "/*")
    input_files.sort()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in input_files:
        basename = os.path.basename(file)
        filename, file_extension = os.path.splitext(basename)

        ff = ffmpy.FFmpeg(
         inputs={file: None},
         outputs={
             output_folder + "/" + filename + "%03d" + file_extension:
                 ['-c:v', vcodec, '-map', '0', '-segment_time', '%d' % split_length, '-f', 'segment',
                  '-reset_timestamps', '1', '-crf', '0', '-g', '9', '-sc_threshold', '0', '-b:v', '0',
                  '-force_key_frames', 'expr:gte(t,n_forced*9)'],
        })
        print(ff.cmd)
        ff.run()


def main():
    parser = OptionParser()

    parser.add_option("-i", "--input",
                        dest = "input_folder",
                        type = "string",
                        action = "store"
                        )
    parser.add_option("-o", "--output",
                        dest = "output_folder",
                        type = "string",
                        action = "store"
                        )
    parser.add_option("-s", "--split-size",
                        dest = "split_length",
                        help = "Split or chunk size in seconds, for example 10",
                        type = "int",
                        action = "store"
                        )
    parser.add_option("-v", "--vcodec",
                      dest = "vcodec",
                      help = "Video codec to use. ",
                      type = "string",
                      default = "copy",
                      action = "store"
                     )

    (options, args) = parser.parse_args()

    if options.input_folder and options.output_folder and options.split_length:
        split_by_seconds(**(options.__dict__))
    else:
        parser.print_help()
        raise SystemExit

if __name__ == '__main__':
    main()