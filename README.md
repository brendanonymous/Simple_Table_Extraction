# Table Extraction with OCR
Simple OpenCV/Tesseract script that uses optical character recognition to extract data tables from a .jpeg or .pdf then writes it to an excel file

# Usage
$ python get-table.py path/to/image/or/pdf/with/table(s)

# Notes
This is my first application of computer vision! It is a POC I'm building for work.
OpenCV works well to preprocess images for optical character recognition with Tesseract.
The input can be a PDF with one to many pages with a different table on each page. The input can also be a JPEG with a table. As of now, the tables in the documents/images should be simple without any whacky characters.
I'm learning on the fly while building this, so at the moment I'm fine tuning everything as I go.
I will add more CV scripts cuz it's very fun!

# Example Input
Random teeeeeeext, blaaaah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah blah........blahhhhhh hello

| Table         | action        | OCR yay  |
| ------------- |:-------------:| --------:|
| row 1 data    | more data     |    1600  |
| row 2 data    | even more data|   12     |
| hello world   |     neat      |    $1    |

___________________________________________________________________________________
(on next page)


|     More      |   TABLE       |  ACTION  |
|:-------------:|:-------------:| --------:|
| fooooooooo    |      data     |    1600  |
| baaaaaaaar    |  daaaaaaata   |   12     |
| table cell    |    hello      |    $1    |
