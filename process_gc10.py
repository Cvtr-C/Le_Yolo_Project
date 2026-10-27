import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_bbox(size, box):
    """Converte coordenadas Pascal VOC (xmin, ymin, xmax, ymax)

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
    """Cria a árvore de diretórios oficial exigida pelo ecossistema Ultralytics."""
    dirs = [
        os.path.join(base_path, "images", "train"),
        os.path.join(base_path, "images", "val"),
        os.path.join(base_path, "labels", "train"),
        os.path.join(base_path, "labels", "val"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def process_gc10_dataset(
    root_dataset_path, output_base, classes_list, split_ratio=0.8, seed=42
):
    """Varre as subpastas de 1 a 10 do GC10-DET para pegar as imagens,

    busca os XMLs correspondentes na pasta 'labels', realiza o split
    e converte para o formato YOLO (.txt).
    """
    print(
        "🚀 Iniciando o processamento do dataset GC10-DET (Imagens nas pastas 1-10, XMLs na pasta labels)..."
    )

    random.seed(seed)

    # Configura diretórios de saída
    img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir = setup_directories(
        output_base
    )

    # Caminho onde ficam os XMLs (pasta 'lable' ou 'labels' dentro de GC10_DET)
    # Ajuste o nome abaixo caso a sua pasta se chame exatamente 'lable' como apareceu no seu 'ls'
    xml_folder_path = os.path.join(
        root_dataset_path, "lable"
    )  # ou "labels" se renomear

    if not os.path.exists(xml_folder_path):
        # Tenta o nome alternativo caso seja 'labels'
        xml_folder_path = os.path.join(root_dataset_path, "labels")

    if not os.path.exists(xml_folder_path):
        print(
            f"❌ Erro Crítico: A pasta de anotações (XMLs) não foi encontrada em {root_dataset_path}!"
        )
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_pairs = []  # Vai armazenar tuplas (caminho_imagem, caminho_xml)

    # Varre as pastas de '1' a '10'
    for class_folder_name in os.listdir(root_dataset_path):
        folder_path = os.path.join(root_dataset_path, class_folder_name)

        # Processa apenas se for diretório e tiver nome numérico de 1 a 10
        if not os.path.isdir(folder_path) or class_folder_name not in [
            str(i) for i in range(1, 11)
        ]:
            continue

        print(f"📂 Lendo imagens da subpasta: {class_folder_name}")

        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                base_name = os.path.splitext(file)[0]
                img_path = os.path.join(folder_path, file)

                # Procura o arquivo XML na pasta central de XMLs ('lable' ou 'labels')
                xml_path = os.path.join(xml_folder_path, base_name + ".xml")

                if os.path.exists(xml_path):
                    all_pairs.append((img_path, xml_path, file))
                else:
                    print(
                        f"⚠️ Aviso: XML não encontrado na pasta de labels para a imagem {file}"
                    )

    if not all_pairs:
        print(
            "❌ Erro: Nenhuma imagem com XML correspondente foi encontrada. Verifique os nomes dos arquivos!"
        )
        return

    # Embaralha os dados para garantir distribuição aleatória i.i.d.
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * split_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"\n📊 Total de pares (Imagem + XML) válidos mapeados: {len(all_pairs)}")
    print(f"📈 Amostras de Treinamento: {len(train_pairs)}")
    print(f"📉 Amostras de Validação: {len(val_pairs)}")

    def process_subset(pairs_list, target_img_dir, target_lbl_dir):
        for img_path, xml_path, filename in pairs_list:
            base_name, _ = os.path.splitext(filename)

            # 1. Copia a imagem para a pasta de destino final (train ou val)
            shutil.copy(img_path, os.path.join(target_img_dir, filename))

            # 2. Faz o parsing do XML e converte para o arquivo .txt do YOLO
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size_elem = root.find("size")
            if size_elem is not None:
                w = int(size_elem.find("width").text)
                h = int(size_elem.find("height").text)
            else:
                w, h = 640, 640  # Padrão de fallback

            txt_filepath = os.path.join(target_lbl_dir, base_name + ".txt")
            with open(txt_filepath, "w") as out_file:
                for obj in root.iter("object"):
                    cls_name = obj.find("name").text

                    if cls_name not in classes_list:
                        try:
                            cls_id = int(cls_name) - 1
                        except ValueError:
                            continue
                    else:
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

    print("\n⏳ Processando e convertendo conjunto de Treinamento...")
    process_subset(train_pairs, img_train_dir, lbl_train_dir)

    print("⏳ Processando e convertendo conjunto de Validação...")
    process_subset(val_pairs, img_val_dir, lbl_val_dir)

    print(
        "\n✅ Sucesso absoluto! Dataset GC10-DET organizado, convertido e pronto para o LeYOLO."
    )


if __name__ == "__main__":
    GC10_CLASSES = [
        "1_chongkong",
        "2_hanfeng",
        "3_yueyawan",
        "4_shuiban",
        "5_youban",
        "6_siban",
        "7_yiwu",
        "8_yahen",
        "9_zhehen",
        "10_yaozhe",
    ]

    DIRETORIO_RAIZ_GC10 = "/home/carlos_vitor/Desktop/DET_GC10/GC10_DET"
    DIRETORIO_SAIDA_YOLO = "/home/carlos_vitor/Desktop/DET_GC10/datasets/gc10_det"

    process_gc10_dataset(
        root_dataset_path=DIRETORIO_RAIZ_GC10,
        output_base=DIRETORIO_SAIDA_YOLO,
        classes_list=GC10_CLASSES,
        split_ratio=0.8,
        seed=42,
    )
