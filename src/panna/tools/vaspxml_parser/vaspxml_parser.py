from io import StringIO
import xml.etree.cElementTree as ET
import os, gzip
import json
import argparse


def main(xml_file, offset, skip, outdir):
    """
    parse vasprun.xml and write relevant info into  panna json.

    xml_file: vasprun.xml file to be parsed
    offset: skip the first 'offset' configurations
    skip: choose configurations every 'skip' steps
    outdir: output directory for panna json files
    """

    if os.path.isdir(outdir):
       outdir = os.path.abspath(outdir)
    else:
       os.mkdir(outdir)
       outdir = os.path.abspath(outdir)

    stream  = StringIO(choose_configs(xml_file, offset, skip))
    panna_json_name = xml_file.split('.xml')[0].split('/')[-1]

    #xml_file_path = os.path.abspath(xml_file)
    #print("---> Parsing {} ...".format(xml_file_path))
    #print("---> Panna json files will be written to \n     {}".format(outdir))

    try:
        i = 1
        for event, elem in ET.iterparse(stream):
            tag = elem.tag
            if tag == "atominfo":
                atom_species = parse_atom_species(elem)
                atom_indices = range(1,len(atom_species)+1)
            elif tag == "varray" and elem.attrib["name"] == "primitive_index":
                atom_indices = [int(v.text.strip()) for v in elem.findall("v")]
            elif tag == "calculation":
                cell_vectors, atom_positions, \
                        total_energy, atom_forces = parse_calculation(elem)
                atoms_info = [list(i) for i in 
                             zip(atom_indices, atom_species, 
                                 atom_positions, atom_forces)]
                write_to_panna_json(xml_file, 
                                    cell_vectors, total_energy, 
                                    atoms_info, outdir.rstrip('/')+"/"+
                                    panna_json_name+"_"+
                                    str(i).zfill(6)+".example")
                i = i+1
            else:
                pass
    except ET.ParseError as err:
        raise err


def choose_configs(xml_file, offset, skip):
    """
        Clean up the input xml file and 
        choose desired configurations.
        
        xml_file: the vasprun.xml file (can be .gz file)
        offset: skip the first "offset" configurations
        skip: choose configurations every "skip" steps
        
        return:
         a xml format string containing 
         the chosen configurations
    """

    if xml_file.split(".")[-1].lower() == "gz":
        f = gzip.open(xml_file,'rt')
    else:
        f = open(xml_file)
        
    run = f.read()
    all_configs = run.split("<calculation>")

    # the calculation setting information
    preamble = all_configs.pop(0)

    # check if the last step is complete
    last_config = all_configs[-1]
    lines = last_config.split("\n")
    complete = any("</calculation>" in line for line in lines)
    if complete:
        index = lines.index(" </calculation>")
        all_configs[-1] = "\n".join(lines[:index+1])
    else:
        print("---> The last configuration is not complete, thus is deleted!")
        del all_configs[-1]

    # choose uncorrelated configurations
    uncorrelated_configs = all_configs[offset::skip]

    to_parse = "<calculation>".join(uncorrelated_configs)
    to_parse = "{}<calculation>{}{}".format(preamble,to_parse,"\n</modeling>")
    
    f.close()
    return to_parse



def parse_atom_species(elem):
    """
        parse the atomic species sequence
        from the preamble of vasprun.xml file
        
        elem: the "atominfo" element just below the root
        return: a list of atomic species
    """
    for a in elem.findall("array"):
        if a.attrib["name"] == "atoms":
            species = [rc.find("c").text.strip()
                              for rc in a.find("set")]
            break
    
    return species



def parse_calculation(elem):
    """
        parse the cell vectors, atom positions,
        total energy and atom forces from an
        "calculation" element.
        
        elem: an "calculation" element 
        returns: 
            cell vectors
            atom positions
            total energy
            atom forces
    """
    # find the total energy of the current configuration
    total_energy = None
    energy = elem.findall('energy')
    for eng in energy[-1]:
        if eng.attrib['name']=='e_wo_entrp':
            total_energy = float(eng.text)
            break

    #finding the forces on the atoms
    forces = None
    varray = elem.findall('varray')
    for v in varray:
        if v.attrib['name'] == 'forces':
            forces = v
    atom_forces = []
    for child in forces:
        atom_forces.append([float(i) for i in child.text.strip().split()])
    
    #finding the cell vectors
    vectors = elem.findall("./structure/crystal/varray/[@name='basis']")[0]
    cell_vectors = []
    for child in vectors:
        cell_vectors.append([float(i) for i in child.text.strip().split()])
    
    #finding the atom positions
    pos = elem.findall("./structure/varray/[@name='positions']")[0]
    atom_positions = []
    for child in pos:
        atom_positions.append([float(i) for i in child.text.strip().split()])
    
    return cell_vectors, atom_positions, total_energy, atom_forces 



def write_to_panna_json(xml_file, cell_vectors, total_energy, 
                        atoms_info, output_filename):
    """
        write the parsed info into panna_json files.
        
        xml_file: vasprun.xml file name
        cell_vectors: lattice vectors 
        total_energy: total energy of the current config
        atoms_info: contain atomic species, indices, atomic positions and atomic forces
        output_filename: panna_json file name to be written
    """

    xml_file_path = os.path.abspath(xml_file)
    key = xml_file.split('.xml')[0].split('/')[-1]
    panna_json = dict()
    panna_json['atomic_position_unit'] = 'crystal'
    panna_json['lattice_vectors'] = cell_vectors
    panna_json['energy'] = [total_energy, "eV"]
    panna_json['atoms'] = atoms_info
    panna_json['source'] = xml_file_path
    panna_json['key'] = key
    panna_json['unit_of_length'] = 'angstrom'
    
    with open(output_filename,'w') as outfile:
        json.dump(panna_json, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="vasprun xml to PANNA json converter")
    parser.add_argument('-i', '--input', type=str, 
                       help='input vasprun.xml file', required=True)
    parser.add_argument('-t', '--offset', type=int,
                       help='offset to the first configuration', required=True)
    parser.add_argument('-p', '--skip', type=int,
                       help='skip every this configurations', required=True)
    parser.add_argument('-o', '--outdir', type=str,
                       help='output directory', required=True)
    
    args = parser.parse_args()
    
    main(args.input, args.offset, args.skip, args.outdir)
