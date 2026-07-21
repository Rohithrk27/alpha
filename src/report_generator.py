import os
from fpdf import FPDF
import pandas as pd
from datetime import datetime, timezone, timedelta

class ReportGenerator:
    """Generates PDF lab reports from models and predictions."""
    
    def __init__(self, output_dir=".temp"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def generate_report(self, eval_results: dict, best_models: dict, predictions_df: pd.DataFrame, img_paths: dict) -> str:
        """
        Creates a PDF report and returns the path to it.
        img_paths: dict with keys like 'actual_vs_predicted', 'shap', 'pca' mapping to temp file paths.
        """
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(0, 15, "Solvent Predictor Lab Report", ln=True, align="C")
        pdf.set_font("Arial", 'I', 10)
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        pdf.cell(0, 10, f"Generated on {datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M')} IST", ln=True, align="C")
        pdf.ln(10)
        
        # Section 1: Top Predictions
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "1. Top Novel Solvent Predictions", ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.multi_cell(0, 8, "The following solvents have been evaluated using the cross-validated Machine Learning engine based on chemical descriptors.")
        pdf.ln(5)
        
        # Dump predictions in a professional table format
        pdf.set_font("Arial", 'B', 10)
        cols_to_print = [c for c in predictions_df.columns if 'Prediction' in c or 'Compatibility' in c or c == 'solvent_name']
        
        # Calculate dynamic column widths based on page size (190mm usable width)
        col_width = 190 / len(cols_to_print)
        
        # Print headers
        for c in cols_to_print:
            header_name = c.replace("_Prediction", "").replace("_", " ")
            pdf.cell(col_width, 10, header_name, border=1, align='C', fill=False)
        pdf.ln()
        
        # Print data rows
        pdf.set_font("Arial", '', 10)
        for idx, row in predictions_df.iterrows():
            for c in cols_to_print:
                val = str(row[c])
                pdf.cell(col_width, 10, val, border=1, align='C')
            pdf.ln()
            
        pdf.ln(10)
        
        # Section 1.5: Top Recommended Structures
        if 'structures' in img_paths and img_paths['structures']:
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Top Recommended Chemical Structures", ln=True)
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 8, "2D structural representations of the highly recommended novel solvents.")
            pdf.ln(5)
            
            struct_imgs = img_paths['structures']
            img_w = 50
            x_start = 15
            y_start = pdf.get_y()
            for i, img_path in enumerate(struct_imgs):
                if i >= 3:
                    break
                if os.path.exists(img_path):
                    pdf.image(img_path, x=x_start + (i * (img_w + 10)), y=y_start, w=img_w)
            
            # Manually advance the Y cursor since pdf.image doesn't
            pdf.set_y(y_start + img_w + 10)
            pdf.ln(5)
            
        # Section 1.8: Experimental vs Predicted Comparison
        if 'combined_bar' in img_paths and os.path.exists(img_paths['combined_bar']):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Experimental Baseline vs. Novel Predictions", ln=True)
            pdf.image(img_paths['combined_bar'], x=15, w=180)
            pdf.ln(10)
        
        # Section 2: Chemical Space
        if 'pca' in img_paths and os.path.exists(img_paths['pca']):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "2. Chemical Space Analysis (PCA)", ln=True)
            pdf.image(img_paths['pca'], x=15, w=180)
            pdf.ln(10)
            
        # Section 3: Model Validation
        if 'actual_vs_predicted' in img_paths and os.path.exists(img_paths['actual_vs_predicted']):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "3. Model Validation (Actual vs Predicted)", ln=True)
            pdf.image(img_paths['actual_vs_predicted'], x=15, w=180)
            pdf.ln(10)
            
        # Section 4: Explainability
        if 'shap' in img_paths and os.path.exists(img_paths['shap']):
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "4. Explainable AI (SHAP)", ln=True)
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 8, "This plot explains which molecular descriptors drove the model's predictions.")
            pdf.image(img_paths['shap'], x=15, w=180)
            
        output_path = os.path.join(self.output_dir, "solvent_lab_report.pdf")
        pdf.output(output_path)
        return output_path
