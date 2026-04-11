import Metashape

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from reconstruction.metashape import build_mesh, build_texture, export_mesh

def main():
    doc = Metashape.Document()
    doc.open(METASHAPE_PATH.format(scanID))

    build_mesh(doc)
    build_texture(doc)
    export_mesh(doc)

if __name__ == "__main__":

    main()