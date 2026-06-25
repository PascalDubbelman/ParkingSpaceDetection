#!/usr/bin/env python3
import xml.etree.ElementTree as ET

def xml_to_txt(label_xml_dir, label_txt_dir):
    '''Convert all .xml files in label_xml_dir to .txt and save them in label_txt_dir.

            Parameters:
                a label_xml_dir (str): Directory containing .xml files.
                b label_txt_dir (str): Directory to save the converted .txt files.

            Returns:
                None
    '''
    label_txt_dir.mkdir(parents=True, exist_ok= True)
    for xml_file in label_xml_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        yolo_lines = []

        # Extract the bounding box values
        for obj in root.findall("object"):
            IMG_W = 1024
            IMG_H = 1024
            class_map = {"parking_space": 0, "1": 0}
            cls_name = obj.find("name").text

            if cls_name in class_map:
                cls_id = class_map[cls_name]

                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                w = xmax - xmin
                h = ymax - ymin
                x_center = xmin + w / 2
                y_center = ymin + h / 2

                yolo_line = f"{cls_id} {x_center/IMG_W:.6f} {y_center/IMG_H:.6f} {w/IMG_W:.6f} {h/IMG_H:.6f}"
                yolo_lines.append(yolo_line)
            else:
                print(f"Warning: Class name '{cls_name}' not found in class_map for file {xml_file.name}.")

        # Convert to text files (necessary for yolo)
        txt_path = label_txt_dir / (xml_file.stem + ".txt")
        txt_path.write_text("\n".join(yolo_lines))
