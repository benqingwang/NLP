import PyPDF2
def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Initialize an empty string to store the extracted text
        text = ''
        
        # Iterate through each page in the PDF
        for page_num in range(len(pdf_reader.pages)):
            # Extract text from the current page
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    
    return text
