import xml.etree.ElementTree as ET

# Sample XML
#<?xml version="1.0"?>
#<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
#  <vtkMultiBlockDataSet>
#    <DataSet index="0" name="Block-00" file="q_criterion_00000001/q_criterion_00000001_0.vtp"/>
#    <DataSet index="1" name="Block-01" file="q_criterion_00000001/q_criterion_00000001_1.vtp"/>
#    <DataSet index="2" name="Block-02" file="q_criterion_00000001/q_criterion_00000001_2.vtp"/>
#    <DataSet index="3" name="Block-03" file="q_criterion_00000001/q_criterion_00000001_3.vtp"/>
#    <DataSet index="4" name="Block-04" file="q_criterion_00000001/q_criterion_00000001_4.vtp"/>
#    <DataSet index="5" name="Block-05" file="q_criterion_00000001/q_criterion_00000001_5.vtp"/>
#    <DataSet index="6" name="Block-06" file="q_criterion_00000001/q_criterion_00000001_6.vtp"/>
#    <DataSet index="7" name="Block-07" file="q_criterion_00000001/q_criterion_00000001_7.vtp"/>
#  </vtkMultiBlockDataSet>
#</VTKFile>

def combine_vtks(files, output_file):

    # Create the root element
    vtk_file = ET.Element('VTKFile', type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian", header_type="UInt32", compressor="vtkZLibDataCompressor")
    vtk_multi_block_data_set = ET.SubElement(vtk_file, 'vtkMultiBlockDataSet')

    # Create the DataSet elements
    for i, file in enumerate(files):
        data_set = ET.SubElement(vtk_multi_block_data_set, 'DataSet', index=str(i), name=f'Block-{str(i).zfill(5)}', file=file)

    # Create the tree
    tree = ET.ElementTree(vtk_file)

    # Write the tree to a file
    tree.write(output_file, encoding='utf-8', xml_declaration=True, method='xml')
