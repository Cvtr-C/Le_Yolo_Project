from ultralytics import YOLO


def main():
    model = YOLO("weights/LeYOLOSmall.pt")
    results = model.train(
        data="yaml/gc10_det.yaml",
        epochs=70,
        imgsz=640,
        batch=16,
        device="cpu",
        workers=4,
        name="leyolo_gc10_run",
    )

    print("🎉 Treinamento concluído com sucesso!")


if __name__ == "__main__":
    main()
