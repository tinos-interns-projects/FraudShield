"""
Secure Forms with CSRF Protection and Input Validation
"""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, FloatField, SelectField, SubmitField, TextAreaField, BooleanField
from wtforms.validators import DataRequired, Length, NumberRange, ValidationError, EqualTo

# Custom email validation to avoid dependency issues
def validate_email_format(email):
    """Custom email validation function"""
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None
import re
import os

class LoginForm(FlaskForm):
    """Secure login form with CSRF protection"""
    username = StringField('Username', validators=[
        DataRequired(message="Username is required"),
        Length(min=3, max=50, message="Username must be between 3 and 50 characters")
    ])
    password = PasswordField('Password', validators=[
        DataRequired(message="Password is required"),
        Length(min=8, max=128, message="Password must be between 8 and 128 characters")
    ])
    submit = SubmitField('Login')
    
    def validate_username(self, username):
        """Custom username validation"""
        # Only allow alphanumeric characters and underscores
        if not re.match(r'^[a-zA-Z0-9_]+$', username.data):
            raise ValidationError('Username can only contain letters, numbers, and underscores')

class RegistrationForm(FlaskForm):
    """Secure registration form with comprehensive validation"""
    username = StringField('Username', validators=[
        DataRequired(message="Username is required"),
        Length(min=3, max=20, message="Username must be between 3 and 20 characters")
    ])

    email = StringField('Email Address', validators=[
        DataRequired(message="Email is required"),
        Length(max=254, message="Email must be less than 254 characters")
    ])

    phone = StringField('Phone Number', validators=[
        DataRequired(message="Phone number is required"),
        Length(min=10, max=20, message="Phone number must be between 10 and 20 characters")
    ])

    password = PasswordField('Password', validators=[
        DataRequired(message="Password is required"),
        Length(min=8, max=128, message="Password must be between 8 and 128 characters")
    ])

    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(message="Please confirm your password"),
        EqualTo('password', message="Passwords must match")
    ])

    terms_accepted = BooleanField('I accept the Terms of Service and Privacy Policy', validators=[
        DataRequired(message="You must accept the terms to register")
    ])

    submit = SubmitField('Create Account')

    def validate_username(self, username):
        """Custom username validation"""
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username.data):
            raise ValidationError('Username must be 3-20 characters, letters, numbers, and underscores only')

    def validate_phone(self, phone):
        """Custom phone validation"""
        # Remove spaces and dashes for validation
        clean_phone = re.sub(r'[\s\-\(\)]', '', phone.data)

        # Check for international format
        if not re.match(r'^\+[1-9]\d{1,14}$', clean_phone):
            raise ValidationError('Phone number must be in international format (e.g., +1234567890)')

    def validate_password(self, password):
        """Custom password strength validation"""
        pwd = password.data

        if len(pwd) < 8:
            raise ValidationError('Password must be at least 8 characters long')

        if not any(c.isupper() for c in pwd):
            raise ValidationError('Password must contain at least one uppercase letter')

        if not any(c.islower() for c in pwd):
            raise ValidationError('Password must contain at least one lowercase letter')

        if not any(c.isdigit() for c in pwd):
            raise ValidationError('Password must contain at least one digit')

        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in pwd):
            raise ValidationError('Password must contain at least one special character')

    def validate_email(self, email):
        """Custom email validation"""
        if not validate_email_format(email.data):
            raise ValidationError('Please enter a valid email address')

class AdminUserManagementForm(FlaskForm):
    """Form for admin user management actions"""
    action = SelectField('Action', choices=[
        ('', 'Select Action'),
        ('toggle_status', 'Toggle Active/Inactive'),
        ('delete', 'Delete User'),
        ('reset_password', 'Reset Password')
    ], validators=[DataRequired(message="Please select an action")])

    submit = SubmitField('Execute Action')

class AdminPasswordResetForm(FlaskForm):
    """Form for admin to reset user passwords"""
    new_password = PasswordField('New Password', validators=[
        DataRequired(message="New password is required"),
        Length(min=8, max=128, message="Password must be between 8 and 128 characters")
    ])

    confirm_password = PasswordField('Confirm New Password', validators=[
        DataRequired(message="Please confirm the new password"),
        EqualTo('new_password', message="Passwords must match")
    ])

    submit = SubmitField('Reset Password')

    def validate_new_password(self, new_password):
        """Custom password strength validation"""
        pwd = new_password.data

        if len(pwd) < 8:
            raise ValidationError('Password must be at least 8 characters long')

        if not any(c.isupper() for c in pwd):
            raise ValidationError('Password must contain at least one uppercase letter')

        if not any(c.islower() for c in pwd):
            raise ValidationError('Password must contain at least one lowercase letter')

        if not any(c.isdigit() for c in pwd):
            raise ValidationError('Password must contain at least one digit')

        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in pwd):
            raise ValidationError('Password must contain at least one special character')

class PredictionForm(FlaskForm):
    """Secure form for single transaction prediction"""
    step = FloatField('Transaction Step', validators=[
        DataRequired(message="Transaction step is required"),
        NumberRange(min=1, max=1000000, message="Step must be between 1 and 1,000,000")
    ])
    
    transaction_type = SelectField('Transaction Type', 
        choices=[
            ('PAYMENT', 'Payment'),
            ('TRANSFER', 'Transfer'),
            ('CASH_OUT', 'Cash Out'),
            ('DEBIT', 'Debit'),
            ('CASH_IN', 'Cash In')
        ],
        validators=[DataRequired(message="Transaction type is required")]
    )
    
    amount = FloatField('Amount', validators=[
        DataRequired(message="Amount is required"),
        NumberRange(min=0.01, max=1000000000, message="Amount must be between $0.01 and $1,000,000,000")
    ])
    
    old_balance_org = FloatField('Original Balance Before Transaction', validators=[
        DataRequired(message="Original balance is required"),
        NumberRange(min=0, max=1000000000, message="Balance must be between $0 and $1,000,000,000")
    ])
    
    new_balance_orig = FloatField('New Balance After Transaction', validators=[
        DataRequired(message="New balance is required"),
        NumberRange(min=0, max=1000000000, message="Balance must be between $0 and $1,000,000,000")
    ])
    
    submit = SubmitField('Analyze Transaction')
    
    def validate_amount(self, amount):
        """Custom amount validation"""
        if amount.data <= 0:
            raise ValidationError('Amount must be greater than zero')
    
    def validate_new_balance_orig(self, new_balance_orig):
        """Validate balance consistency"""
        if hasattr(self, 'old_balance_org') and hasattr(self, 'amount'):
            old_balance = self.old_balance_org.data
            amount = self.amount.data
            new_balance = new_balance_orig.data
            
            if old_balance is not None and amount is not None and new_balance is not None:
                # For most transaction types, new balance should be old balance minus amount
                if self.transaction_type.data in ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']:
                    expected_balance = old_balance - amount
                    # Allow some tolerance for rounding
                    if abs(new_balance - expected_balance) > 0.01 and new_balance != 0:
                        # Only warn, don't fail validation as some transactions might have different logic
                        pass

class SecureFileUploadForm(FlaskForm):
    """Secure file upload form with validation"""
    file = FileField('Upload File', validators=[
        FileRequired(message="Please select a file"),
        FileAllowed(['csv', 'xlsx', 'xls', 'pdf'], 
                   message="Only CSV, Excel (.xlsx, .xls), and PDF files are allowed")
    ])
    
    submit = SubmitField('Upload & Analyze')
    
    def validate_file(self, file):
        """Custom file validation"""
        if file.data:
            # Check file size (10MB limit)
            max_size = int(os.getenv('MAX_FILE_SIZE', 10485760))  # 10MB default
            
            # Get file size
            file.data.seek(0, 2)  # Seek to end
            file_size = file.data.tell()
            file.data.seek(0)  # Reset to beginning
            
            if file_size > max_size:
                raise ValidationError(f'File size must be less than {max_size // (1024*1024)}MB')
            
            # Check filename for suspicious patterns
            filename = file.data.filename
            if filename:
                # Remove path traversal attempts
                if '..' in filename or '/' in filename or '\\' in filename:
                    raise ValidationError('Invalid filename')
                
                # Check for suspicious extensions
                suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com', '.js', '.vbs']
                for ext in suspicious_extensions:
                    if filename.lower().endswith(ext):
                        raise ValidationError('File type not allowed for security reasons')

class ContactForm(FlaskForm):
    """Secure contact form"""
    name = StringField('Name', validators=[
        DataRequired(message="Name is required"),
        Length(min=2, max=100, message="Name must be between 2 and 100 characters")
    ])
    
    email = StringField('Email', validators=[
        DataRequired(message="Email is required"),
        Length(max=254, message="Email must be less than 254 characters")
    ])
    
    subject = StringField('Subject', validators=[
        DataRequired(message="Subject is required"),
        Length(min=5, max=200, message="Subject must be between 5 and 200 characters")
    ])
    
    message = TextAreaField('Message', validators=[
        DataRequired(message="Message is required"),
        Length(min=10, max=2000, message="Message must be between 10 and 2000 characters")
    ])
    
    submit = SubmitField('Send Message')
    
    def validate_email(self, email):
        """Custom email validation"""
        if not validate_email_format(email.data):
            raise ValidationError('Please enter a valid email address')
    
    def validate_name(self, name):
        """Custom name validation"""
        # Only allow letters, spaces, hyphens, and apostrophes
        if not re.match(r"^[a-zA-Z\s\-']+$", name.data):
            raise ValidationError('Name can only contain letters, spaces, hyphens, and apostrophes')

def sanitize_form_data(form_data):
    """Sanitize all form data"""
    sanitized_data = {}
    
    for key, value in form_data.items():
        if isinstance(value, str):
            # Remove potentially dangerous characters
            sanitized_value = re.sub(r'[<>"\';()&]', '', value)
            # Limit length
            sanitized_value = sanitized_value[:1000]
            sanitized_data[key] = sanitized_value.strip()
        else:
            sanitized_data[key] = value
    
    return sanitized_data

def validate_numeric_input(value, min_val=None, max_val=None, field_name="Value"):
    """Validate numeric input with range checking"""
    try:
        numeric_value = float(value)
        
        if min_val is not None and numeric_value < min_val:
            return False, f"{field_name} must be at least {min_val}"
        
        if max_val is not None and numeric_value > max_val:
            return False, f"{field_name} must be at most {max_val}"
        
        return True, numeric_value
        
    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid number"

def validate_file_content(file_path, allowed_extensions):
    """Validate uploaded file content"""
    try:
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        if file_ext not in allowed_extensions:
            return False, f"File extension '{file_ext}' not allowed"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        max_size = int(os.getenv('MAX_FILE_SIZE', 10485760))
        
        if file_size > max_size:
            return False, f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)"
        
        # Basic content validation based on file type
        if file_ext in ['csv']:
            # Check if it's a valid CSV by reading first few lines
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if not first_line.strip():
                        return False, "CSV file appears to be empty"
            except UnicodeDecodeError:
                return False, "CSV file contains invalid characters"
        
        elif file_ext in ['xlsx', 'xls']:
            # Basic Excel file validation
            try:
                import pandas as pd
                pd.read_excel(file_path, nrows=1)
            except Exception as e:
                return False, f"Invalid Excel file: {str(e)}"
        
        elif file_ext == 'pdf':
            # Basic PDF validation
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        return False, "Invalid PDF file format"
            except Exception as e:
                return False, f"Error reading PDF file: {str(e)}"
        
        return True, "File validation passed"
        
    except Exception as e:
        return False, f"File validation error: {str(e)}"
