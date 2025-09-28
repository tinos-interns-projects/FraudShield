#!/usr/bin/env python3
"""
Create a test PDF file with tabular data for testing PDF upload functionality
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
import pandas as pd

def create_test_pdf():
    """Create a test PDF with fraud detection sample data"""
    
    # Sample data
    data = {
        'step': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT', 'CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT'],
        'amount': [1000.0, 5000.0, 2500.0, 750.0, 10000.0, 300.0, 7500.0, 1200.0, 15000.0, 500.0],
        'oldbalanceOrg': [5000.0, 5000.0, 2500.0, 3000.0, 10000.0, 1000.0, 7500.0, 5000.0, 15000.0, 2000.0],
        'newbalanceOrig': [4000.0, 0.0, 0.0, 2250.0, 0.0, 700.0, 0.0, 3800.0, 0.0, 1500.0]
    }
    
    df = pd.DataFrame(data)
    
    # Create PDF
    doc = SimpleDocTemplate("static/test_fraud_data.pdf", pagesize=letter)
    story = []
    
    # Title
    styles = getSampleStyleSheet()
    title = Paragraph("Fraud Detection Test Dataset", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 20))
    
    # Description
    desc = Paragraph("This PDF contains sample transaction data for testing the fraud detection system's PDF upload capability.", styles['Normal'])
    story.append(desc)
    story.append(Spacer(1, 20))
    
    # Create table data
    table_data = [['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
    
    for _, row in df.iterrows():
        table_data.append([
            str(row['step']),
            str(row['type']),
            str(row['amount']),
            str(row['oldbalanceOrg']),
            str(row['newbalanceOrig'])
        ])
    
    # Create table
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    
    # Build PDF
    doc.build(story)
    print("✅ Test PDF created: static/test_fraud_data.pdf")

if __name__ == "__main__":
    create_test_pdf()
