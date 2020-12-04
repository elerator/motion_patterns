from pathlib import Path
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Preprocessing script')
    parser.add_argument('-folder', help="Clears all files with the specified ending from the folder")
    parser.add_argument('-file_ending', help="All files with this ending will be cleared")
    args = parser.parse_args()
    
    if os.path.isdir(args.folder):
      for f in os.listdir(args.folder):
        if f.endswith(args.file_ending):
          print("Deleting " + f)
          os.remove(os.path.join(args.folder,f))
    else:
      print("Not a folder")
  