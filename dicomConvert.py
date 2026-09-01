#Adapted from https://github.com/dennyglee/dicom-to-png
import os
import png
import dicom
import argparse


def ct_to_png(ct_file, png_file):
    """ Function to convert from a DICOM image to png

        @param ct_file: An opened file like object to read te dicom data
        @param png_file: An opened file like object to write the png data
    """

    # Extracting data from the ct file
    plan = dicom.read_file(ct_file)
    shape = plan.pixel_array.shape

    image_2d = []
    max_val = 0
    for row in plan.pixel_array:
        pixels = []
        for col in row:
            pixels.append(col)
            if col > max_val: max_val = col
        image_2d.append(pixels)

    # Rescaling grey scale between 0-255
    image_2d_scaled = []
    for row in image_2d:
        row_scaled = []
        for col in row:
            col_scaled = int((float(col) / float(max_val)) * 255.0)
            row_scaled.append(col_scaled)
        image_2d_scaled.append(row_scaled)

    # Writing the PNG file
    w = png.Writer(shape[0], shape[1], greyscale=True)
    w.write(png_file, image_2d_scaled)


def convert_file(ct_file_path, png_file_path):
    """ Function to convert a CT binary file to a
        PNG image file.

        @param ct_file_path: Full path to the ct file
        @param png_file_path: Fill path to the png file
    """

    # Making sure that the ct file exists
    if not os.path.exists(ct_file_path):
        raise Exception('File "%s" does not exists' % ct_file_path)

    # Making sure the png file does not exist
    if os.path.exists(png_file_path):
        raise Exception('File "%s" already exists' % png_file_path)

    ct_file = open(ct_file_path, 'rb')
    png_file = open(png_file_path, 'wb')

    ct_to_png(ct_file, png_file)

    png_file.close()


def convert_folder(ct_folder, png_folder):
    """ Convert all ct files in a folder to png files
        in a destination folder
    """

    # Create the folder for the pnd directory structure
    os.makedirs(png_folder)

    # Recursively traverse all sub-folders in the path
    for ct_sub_folder, subdirs, files in os.walk(ct_folder):
        for ct_file in os.listdir(ct_sub_folder):
            ct_file_path = os.path.join(ct_sub_folder, ct_file)

            # Make sure path is an actual file
            if os.path.isfile(ct_file_path):

                # Replicate the original file structure
                rel_path = os.path.relpath(ct_sub_folder, ct_folder)
                png_folder_path = os.path.join(png_folder, rel_path)
                if not os.path.exists(png_folder_path):
                    os.makedirs(png_folder_path)
                png_file_path = os.path.join(png_folder_path, '%s.png' % ct_file)

                try:
                    # Convert the actual file
                    convert_file(ct_file_path, png_file_path)
                    print( "SUCCESS", ct_file_path, '-->', png_file_path)
                except Exception as e:
                    print ("FAIL", ct_file_path, '-->', png_file_path, ':', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a dicom ct file to png")
    parser.add_argument('-f', action='store_true')
    parser.add_argument('dicom_path', help='Full path to the ct file')
    parser.add_argument('png_path', help='Full path to the generated png file')

    args = parser.parse_args()
    print (args)
    if args.f:
        convert_folder(args.dicom_path, args.png_path)
    else:
        convert_file(args.dicom_path, args.png_path)