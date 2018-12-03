import os
from optparse import OptionParser
import ffmpy
import glob
from random import shuffle

def split_by_seconds(input_folder, output_file, vcodec="copy", **kwargs):

    input_files = glob.glob(input_folder + "/*")
    shuffle(input_files)

    inputs = {}
    filter_options = ""

    for idx, file in enumerate(input_files):
        basename = os.path.basename(file)
        filename, file_extension = os.path.splitext(basename)

        inputs[file] = None

        filter_options += "[%d:v:0]" % idx

    filter_options += "concat=n=%d:v=1[outv]" % len(input_files)

    ff = ffmpy.FFmpeg(
     inputs=inputs,
     outputs={
         output_file :
             ['-c:v', vcodec, '-filter_complex', filter_options, '-map', '[outv]', '-b:v', '0', '-crf', '20'],
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
                        dest = "output_file",
                        type = "string",
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

    if options.input_folder and options.output_file:
        split_by_seconds(**(options.__dict__))
    else:
        parser.print_help()
        raise SystemExit

if __name__ == '__main__':
    main()