# Simple_Table_Extraction
Simple OpenCV/Tesseract script that uses optical character recognition to extract data tables from a .jpeg or .pdf then writes it to an excel file

# Usage
$ python get-table.py path/to/image/or/pdf/with/table(s)

# Notes
This is my first application of computer vision! It is a POC I'm building for work.
OpenCV works well to preprocess images for optical character recognition with Tesseract.
The input can be a PDF with one to many pages with a different table on each page. The input can also be a JPEG with a table. As of now, the tables in the documents/images should be simple without any whacky characters.
I'm learning on the fly while building this, so at the moment I'm fine tuning everything as I go.
I will add more CV scripts cuz it's very fun!
