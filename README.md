# OCR
    This project tries to train a hebrew OCR for mainly documents
    Steps:
     - install tesseract: https://github.com/UB-Mannheim/tesseract/wiki
     - make sure it is in path (for windows)
     - install tesstrain (includes installing tesserect): https://github.com/tesseract-ocr/tesstrain/tree/main
     - run pipeline notebook
     
     WIP:
     - infrastructure:
        - pipeline evaluation is incomplete - fix
        - remove pipeline train dependency on default folders
     - evaluation:
        - acquire different documents (prefebly PDFs) to check model
        - compare trained model to online's tesseract best
        - experiments:
            - eval on unknown fonts
            - eval on unknown words
            - typos
        - augmentations:
            - in pdf's level:
                - fonts
                - sizes
                - color (text and background)
                - punctuation (pisuk not nikud)
                - larger vocabulary
            - in image level:
                - noise (gaussian, S&P)
                - printer artifacts
                - brightness changes
                - roatations (small)
                
                
                
