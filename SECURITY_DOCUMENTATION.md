# 🔒 Fraud Detection System - Security & User Management Documentation

## 🎯 Overview
This document outlines the comprehensive security features and user management system implemented in the Fraud Detection System, including user registration, authentication, and admin control panel.

## 🔐 Authentication System

### User Registration (Sign-Up)
- **URL:** `/register`
- **Features:**
  - Unique username validation (3-20 characters, alphanumeric + underscore)
  - Valid email address with duplicate checking
  - International phone number format (+country code)
  - Strong password requirements
  - Terms of service acceptance
  - Real-time password strength validation

### User Login
- **URL:** `/login`
- **Features:**
  - Secure username/password authentication
  - Account lockout after 5 failed attempts (30-minute lockout)
  - Session management with 1-hour timeout
  - Rate limiting (5 attempts per minute)
  - Security event logging

### Password Requirements
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)

## 👥 User Management

### User Data Storage
- **Storage:** JSON file (`users.json`)
- **Security:** Passwords hashed with bcrypt
- **Backup:** Automatic file backup on changes

### User Roles
1. **Admin Users:**
   - Full system access
   - User management capabilities
   - System monitoring
   - Cannot be disabled or deleted

2. **Regular Users:**
   - Standard application access
   - Fraud detection features
   - File upload capabilities
   - Activity tracking

### Default Accounts
```
Admin Account:
- Username: admin
- Password: SecurePassword123!
- Email: admin@frauddetection.com
- Phone: +1-555-0001

Demo Account:
- Username: demo
- Password: DemoUser456!
- Email: demo@frauddetection.com
- Phone: +1-555-0002
```

## 🛡️ Admin Control Panel

### Access
- **URL:** `/admin`
- **Requirements:** Admin role + authentication
- **Rate Limiting:** 30 requests per minute

### Features

#### 1. User Statistics Dashboard
- Total users count
- Active/inactive users
- Recent login activity
- Total predictions made
- Total file uploads
- Visual charts and graphs

#### 2. User Management Table
- **Search & Filter:**
  - Search by username/email
  - Filter by status (active/inactive)
  - Filter by role (admin/user)

- **User Information Display:**
  - Username and user ID
  - Email and phone number
  - Role and status badges
  - Last login timestamp
  - Activity statistics (logins, predictions, uploads)

#### 3. Admin Actions
- **Toggle User Status:** Enable/disable user accounts
- **Reset Password:** Set new passwords for users
- **Delete Users:** Remove users from system (except admins)
- **Activity Monitoring:** Track user actions

### Admin API Endpoints
- `POST /admin/toggle_user/<user_id>` - Toggle user active status
- `DELETE /admin/delete_user/<user_id>` - Delete user account
- `POST /admin/reset_password/<user_id>` - Reset user password

## 🔒 Security Features

### 1. Input Validation & Sanitization
- **Form Validation:** Server-side and client-side validation
- **CSRF Protection:** All forms protected with CSRF tokens
- **Input Sanitization:** Removal of dangerous characters
- **SQL Injection Prevention:** Parameterized queries and input cleaning

### 2. File Upload Security
- **File Type Validation:** Only CSV, Excel, PDF allowed
- **File Size Limits:** Maximum 10MB per file
- **Content Validation:** Deep file structure verification
- **Malware Prevention:** Suspicious pattern detection
- **Automatic Cleanup:** Temporary files removed after processing

### 3. Rate Limiting
- **Login Attempts:** 5 per minute
- **Registration:** 3 per minute
- **API Requests:** 100 per hour (default)
- **Predictions:** 20 per minute
- **File Uploads:** 10 per minute
- **Admin Panel:** 30 per minute

### 4. Session Security
- **Session Timeout:** 1 hour automatic logout
- **Secure Cookies:** HttpOnly and SameSite protection
- **Session Invalidation:** Proper logout handling
- **CSRF Protection:** Cross-site request forgery prevention

### 5. Error Handling
- **Safe Error Messages:** No sensitive information exposed
- **Error Tracking:** Unique error IDs for monitoring
- **Security Logging:** All security events logged
- **Attack Detection:** Potential attack pattern recognition

## 📊 Activity Tracking

### User Activity Metrics
- **Total Logins:** Number of successful login attempts
- **Total Predictions:** Number of fraud predictions made
- **Total Uploads:** Number of files uploaded for analysis
- **Last Login:** Timestamp of most recent login
- **Account Creation:** User registration timestamp

### Security Event Logging
- Login attempts (success/failure)
- User registration events
- File upload attempts
- Admin actions (user management)
- Potential security threats
- Account lockouts and unlocks

## 🚀 Getting Started

### 1. Start the Application
```bash
python run_secure_app.py
```

### 2. Access Points
- **Home Page:** http://127.0.0.1:5000/
- **Registration:** http://127.0.0.1:5000/register
- **Login:** http://127.0.0.1:5000/login
- **Admin Panel:** http://127.0.0.1:5000/admin (admin only)

### 3. First-Time Setup
1. Start the application
2. Visit the registration page to create new accounts
3. Login with admin credentials to access admin panel
4. Manage users and monitor system activity

## 🔧 Configuration

### Environment Variables (.env)
```
SECRET_KEY=fraud_detection_super_secret_key_2024
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_EXTENSIONS=csv,xlsx,xls,pdf
SESSION_TIMEOUT=3600    # 1 hour
DEFAULT_RATE_LIMIT=100 per hour
ADMIN_USERNAME=admin
ADMIN_PASSWORD=SecurePassword123!
DEMO_USERNAME=demo
DEMO_PASSWORD=DemoUser456!
```

### Security Best Practices
1. **Change Default Passwords:** Update admin and demo passwords
2. **Use HTTPS:** Enable SSL/TLS in production
3. **Regular Backups:** Backup user data and logs
4. **Monitor Logs:** Review security logs regularly
5. **Update Dependencies:** Keep packages up to date

## 📋 API Documentation

### Registration API
```
POST /register
Content-Type: application/x-www-form-urlencoded

username=newuser&email=user@example.com&phone=+1234567890&password=SecurePass123!&confirm_password=SecurePass123!&terms_accepted=on
```

### Login API
```
POST /login
Content-Type: application/x-www-form-urlencoded

username=admin&password=SecurePassword123!
```

### Admin User Management
```
POST /admin/toggle_user/<user_id>
Authorization: Admin session required

DELETE /admin/delete_user/<user_id>
Authorization: Admin session required

POST /admin/reset_password/<user_id>
Content-Type: application/json
Authorization: Admin session required

{
  "new_password": "NewSecurePassword123!"
}
```

## 🛠️ Troubleshooting

### Common Issues
1. **Registration Fails:** Check password requirements and unique constraints
2. **Login Issues:** Verify credentials and account status
3. **Admin Access Denied:** Ensure user has admin role
4. **File Upload Errors:** Check file size and format restrictions
5. **Session Timeout:** Re-login after 1 hour of inactivity

### Log Files
- `fraud_detection.log` - General application logs
- `security.log` - Security-specific events
- `users.json` - User database (encrypted passwords)

## 🎯 Features Summary

✅ **User Registration** with comprehensive validation
✅ **Secure Authentication** with account lockout protection
✅ **Admin Control Panel** for user management
✅ **Activity Tracking** and statistics
✅ **Security Logging** and monitoring
✅ **Rate Limiting** and DDoS protection
✅ **File Upload Security** with validation
✅ **Session Management** with timeout
✅ **CSRF Protection** on all forms
✅ **Input Sanitization** and validation

The system is now enterprise-ready with comprehensive security measures and user management capabilities!
