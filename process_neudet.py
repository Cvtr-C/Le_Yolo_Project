import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_bbox(size, box):
    """
    Converte coordenadas de caixa delimitadora Pascal VOC (xmin, ymin, xmax, ymax)
    para o formato YOLO normalizado (x_center, y_center, width, height).
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0 - 1.0
    y = (box[1] + box[3]) / 2.0 - 1.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)


def setup_directories(base_path):
    """
    Cria a árvore de diretórios oficial exigida pelo ecossistema Ultralytics.
    """
    dirs = [
        os.path.join(base_path, "images", "train"),
        os.path.join(base_path, "images", "val"),
        os.path.join(base_path, "labels", "train"),
        os.path.join(base_path, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def process_and_split_dataset(
    img_src, xml_src, output_base, classes_list, split_ratio=0.8, seed=42
):
    """
    Executa o particionamento estratificado/aleatório do dataset e a conversão simultânea
    dos arquivos XML (Pascal VOC) para arquivos de anotação `.txt` (formato YOLO).
    """
    print("🚀 Iniciando o pipeline unificado de particionamento e conversão...")

    random.seed(seed)

    # Configura diretórios de saída
    img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir = setup_directories(
        output_base
    )

    # Coleta todas as imagens válidas
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [
        f
        for f in os.listdir(img_src)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]

    # Embaralha com semente fixa para garantir reprodutibilidade e independência estatística
    random.shuffle(all_images)

    split_idx = int(len(all_images) * split_ratio)
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]

    print(f"📊 Total de imagens mapeadas: {len(all_images)}")
    print(f"📈 Amostras de Treinamento: {len(train_files)}")
    print(f"📉 Amostras de Validação: {len(val_files)}")

    def process_subset(file_list, target_img_dir, target_lbl_dir):
        missing_xmls = 0
        for filename in file_list:
            base_name, _ = os.path.splitext(filename)
            xml_filename = base_name + ".xml"

            src_img_path = os.path.join(img_src, filename)
            src_xml_path = os.path.join(xml_src, xml_filename)

            if not os.path.exists(src_xml_path):
                print(f"⚠️ Aviso: Arquivo XML não encontrado para a imagem {filename}")
                missing_xmls += 1
                continue

            # 1. Copia a imagem para o destino final (train ou val)
            shutil.copy(src_img_path, os.path.join(target_img_dir, filename))

            # 2. Faz o parsing do XML e converte diretamente para o arquivo .txt do YOLO
            tree = ET.parse(src_xml_path)
            root = tree.getroot()

            size_elem = root.find("size")
            if size_elem is not None:
                w = int(size_elem.find("width").text)
                h = int(size_elem.find("height").text)
            else:
                # Fallback caso a tag size esteja ausente (lê dimensões da imagem se necessário)
                w, h = 640, 640

            txt_filepath = os.path.join(target_lbl_dir, base_name + ".txt")
            with open(txt_filepath, "w") as out_file:
                for obj in root.iter("object"):
                    cls_name = obj.find("name").text
                    if cls_name not in classes_list:
                        continue
                    cls_id = classes_list.index(cls_name)

                    xmlbox = obj.find("bndbox")
                    b = (
                        float(xmlbox.find("xmin").text),
                        float(xmlbox.find("ymin").text),
                        float(xmlbox.find("xmax").text),
                        float(xmlbox.find("ymax").text),
                    )

                    bb = convert_bbox((w, h), b)
                    out_file.write(
                        f"{cls_id} {' '.join([f'{coord:.6f}' for coord in bb])}\n"
                    )

        return missing_xmls

    print("\n⏳ Processando e convertendo conjunto de Treinamento...")
    missing_train = process_subset(train_files, img_train_dir, lbl_train_dir)

    print("⏳ Processando e convertendo conjunto de Validação...")
    missing_val = process_subset(val_files, img_val_dir, lbl_val_dir)

    if missing_train == 0 and missing_val == 0:
        print(
            "\n✅ Sucesso absoluto! Dataset particionado e convertido pronto para o LeYOLO."
        )
    else:
        print(
            f"\n⚠️ Processo concluído com {missing_train + missing_val} inconsistências de XMLs ausentes."
        )


if __name__ == "__main__":
    # Definição das classes oficiais do NEU-DET
    NEU_CLASSES = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]

    # Caminhos locais customizados para o seu ambiente de trabalho
    DIRETORIO_IMAGENS = "/home/carlos_vitor/Desktop/DET_GC10/NEU_DET/IMAGES"
    DIRETORIO_XMLS = "/home/carlos_vitor/Desktop/DET_GC10/NEU_DET/ANNOTATIONS"
    DIRETORIO_SAIDA_YOLO = "/home/carlos_vitor/Desktop/DET_GC10/datasets/neu_det"

    process_and_split_dataset(
        img_src=DIRETORIO_IMAGENS,
        xml_src=DIRETORIO_XMLS,
        output_base=DIRETORIO_SAIDA_YOLO,
        classes_list=NEU_CLASSES,
        split_ratio=0.8,
        seed=42,
    )
