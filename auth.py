"""
Authentication and User Management Module
Handles user authentication, session management, and security
"""

import os
import bcrypt
import json
import re
from flask_login import UserMixin
from datetime import datetime, timedelta
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class User(UserMixin):
    """Enhanced User class for Flask-Login with registration support"""

    def __init__(self, id, username, password_hash, email=None, phone=None, role='user', active=True):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.email = email
        self.phone = phone
        self.role = role
        self._is_active = active
        self.last_login = None
        self.login_attempts = 0
        self.locked_until = None
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.total_logins = 0
        self.total_predictions = 0
        self.total_uploads = 0

    @property
    def is_active(self):
        """Check if user is active"""
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        """Set user active status"""
        self._is_active = value
    
    def check_password(self, password):
        """Check if provided password matches the hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password check error: {e}")
            return False
    
    def is_admin(self):
        """Check if user has admin role"""
        return self.role == 'admin'

    def is_developer_admin(self):
        """Check if user is a developer/system admin with full access"""
        return self.role == 'admin' and self.username in ['admin', 'developer', 'sysadmin']
    
    def is_locked(self):
        """Check if account is locked due to failed attempts"""
        if self.locked_until and datetime.now() < self.locked_until:
            return True
        return False
    
    def lock_account(self, duration_minutes=30):
        """Lock account for specified duration"""
        self.locked_until = datetime.now() + timedelta(minutes=duration_minutes)
        logger.warning(f"Account {self.username} locked until {self.locked_until}")
    
    def unlock_account(self):
        """Unlock account and reset login attempts"""
        self.locked_until = None
        self.login_attempts = 0
        logger.info(f"Account {self.username} unlocked")
    
    def record_login_attempt(self, success=False):
        """Record login attempt"""
        if success:
            self.login_attempts = 0
            self.last_login = datetime.now().isoformat()
            self.total_logins += 1
            self.updated_at = datetime.now().isoformat()
            logger.info(f"Successful login for {self.username}")
        else:
            self.login_attempts += 1
            logger.warning(f"Failed login attempt for {self.username} (attempt {self.login_attempts})")

            # Lock account after 5 failed attempts
            if self.login_attempts >= 5:
                self.lock_account()

    def record_prediction(self):
        """Record a prediction made by user"""
        self.total_predictions += 1
        self.updated_at = datetime.now().isoformat()

    def record_upload(self):
        """Record a file upload by user"""
        self.total_uploads += 1
        self.updated_at = datetime.now().isoformat()

    def to_dict(self):
        """Convert user to dictionary for JSON storage"""
        return {
            'id': self.id,
            'username': self.username,
            'password_hash': self.password_hash,
            'email': self.email,
            'phone': self.phone,
            'role': self.role,
            'is_active': self._is_active,
            'last_login': self.last_login,
            'login_attempts': self.login_attempts,
            'locked_until': self.locked_until.isoformat() if self.locked_until else None,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'total_logins': self.total_logins,
            'total_predictions': self.total_predictions,
            'total_uploads': self.total_uploads
        }

    @classmethod
    def from_dict(cls, data):
        """Create user from dictionary"""
        user = cls(
            id=data['id'],
            username=data['username'],
            password_hash=data['password_hash'],
            email=data.get('email'),
            phone=data.get('phone'),
            role=data.get('role', 'user'),
            active=data.get('is_active', True)
        )
        user.last_login = data.get('last_login')
        user.login_attempts = data.get('login_attempts', 0)
        user.locked_until = datetime.fromisoformat(data['locked_until']) if data.get('locked_until') else None
        user.created_at = data.get('created_at', datetime.now().isoformat())
        user.updated_at = data.get('updated_at', datetime.now().isoformat())
        user.total_logins = data.get('total_logins', 0)
        user.total_predictions = data.get('total_predictions', 0)
        user.total_uploads = data.get('total_uploads', 0)
        return user

class UserManager:
    """Enhanced user management with JSON storage and registration"""

    def __init__(self, users_file='users.json'):
        self.users_file = users_file
        self.users = {}
        self._load_users_from_file()
        self._initialize_default_users()

    def _load_users_from_file(self):
        """Load users from JSON file"""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    users_data = json.load(f)
                    for user_id, user_data in users_data.items():
                        self.users[user_id] = User.from_dict(user_data)
                logger.info(f"Loaded {len(self.users)} users from {self.users_file}")
            else:
                logger.info(f"Users file {self.users_file} not found, starting with empty user database")
        except Exception as e:
            logger.error(f"Error loading users from file: {e}")
            self.users = {}

    def _save_users_to_file(self):
        """Save users to JSON file"""
        try:
            users_data = {}
            for user_id, user in self.users.items():
                users_data[user_id] = user.to_dict()

            with open(self.users_file, 'w') as f:
                json.dump(users_data, f, indent=2)
            logger.info(f"Saved {len(self.users)} users to {self.users_file}")
        except Exception as e:
            logger.error(f"Error saving users to file: {e}")

    def _initialize_default_users(self):
        """Initialize default users from environment variables if they don't exist"""
        try:
            # Admin user
            admin_username = os.getenv('ADMIN_USERNAME', 'admin')
            if not any(user.username == admin_username for user in self.users.values()):
                admin_password = os.getenv('ADMIN_PASSWORD', 'SecurePassword123!')
                admin_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

                self.users['admin'] = User(
                    id='admin',
                    username=admin_username,
                    password_hash=admin_hash,
                    email='admin@frauddetection.com',
                    phone='+1-555-0001',
                    role='admin'
                )

            # Demo user
            demo_username = os.getenv('DEMO_USERNAME', 'demo')
            if not any(user.username == demo_username for user in self.users.values()):
                demo_password = os.getenv('DEMO_PASSWORD', 'DemoUser456!')
                demo_hash = bcrypt.hashpw(demo_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

                self.users['demo'] = User(
                    id='demo',
                    username=demo_username,
                    password_hash=demo_hash,
                    email='demo@frauddetection.com',
                    phone='+1-555-0002',
                    role='user'
                )

            # Save users if any were added
            self._save_users_to_file()
            logger.info("Default users initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing default users: {e}")
    
    def get_user(self, user_id):
        """Get user by ID"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username):
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def authenticate_user(self, username, password):
        """Authenticate user with username and password"""
        try:
            user = self.get_user_by_username(username)
            
            if not user:
                logger.warning(f"Authentication failed: User {username} not found")
                return None
            
            if user.is_locked():
                logger.warning(f"Authentication failed: User {username} is locked")
                return None
            
            if user.check_password(password):
                user.record_login_attempt(success=True)
                return user
            else:
                user.record_login_attempt(success=False)
                return None
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def register_user(self, username, email, phone, password):
        """Register a new user with validation"""
        try:
            # Validate input
            validation_result = self.validate_registration_data(username, email, phone, password)
            if not validation_result[0]:
                return validation_result

            # Create new user
            user_id = str(uuid.uuid4())
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            new_user = User(
                id=user_id,
                username=username,
                password_hash=password_hash,
                email=email,
                phone=phone,
                role='user'
            )

            self.users[user_id] = new_user
            self._save_users_to_file()

            logger.info(f"New user {username} registered successfully")
            return True, "User registered successfully"

        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return False, f"Registration failed: {str(e)}"

    def validate_registration_data(self, username, email, phone, password):
        """Validate registration data"""
        try:
            # Check for duplicate username
            if any(user.username.lower() == username.lower() for user in self.users.values()):
                return False, "Username already exists"

            # Check for duplicate email
            if any(user.email and user.email.lower() == email.lower() for user in self.users.values()):
                return False, "Email already registered"

            # Validate username format
            if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
                return False, "Username must be 3-20 characters, letters, numbers, and underscores only"

            # Validate email format
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                return False, "Invalid email format"

            # Validate phone number format
            phone_pattern = r'^\+[1-9]\d{1,14}$'  # International format
            if not re.match(phone_pattern, phone):
                return False, "Phone number must be in international format (+country code)"

            # Validate password strength
            password_validation = validate_password_strength(password)
            if not password_validation[0]:
                return password_validation

            return True, "Validation passed"

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, "Validation failed"

    def update_user_activity(self, user_id, activity_type):
        """Update user activity statistics"""
        try:
            user = self.users.get(user_id)
            if user:
                if activity_type == 'prediction':
                    user.record_prediction()
                elif activity_type == 'upload':
                    user.record_upload()

                self._save_users_to_file()
                return True
        except Exception as e:
            logger.error(f"Error updating user activity: {e}")
        return False

    def toggle_user_status(self, user_id):
        """Toggle user active/inactive status (admin only)"""
        try:
            user = self.users.get(user_id)
            if user and user.role != 'admin':  # Don't disable admin users
                user.is_active = not user.is_active
                user.updated_at = datetime.now().isoformat()
                self._save_users_to_file()
                logger.info(f"User {user.username} status changed to {'active' if user.is_active else 'inactive'}")
                return True, f"User {'activated' if user.is_active else 'deactivated'} successfully"
        except Exception as e:
            logger.error(f"Error toggling user status: {e}")
        return False, "Failed to update user status"

    def delete_user(self, user_id):
        """Delete a user (admin only)"""
        try:
            user = self.users.get(user_id)
            if user and user.role != 'admin':  # Don't delete admin users
                username = user.username
                del self.users[user_id]
                self._save_users_to_file()
                logger.info(f"User {username} deleted successfully")
                return True, "User deleted successfully"
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
        return False, "Failed to delete user"

    def get_user_statistics(self):
        """Get user statistics for admin panel"""
        try:
            total_users = len(self.users)
            active_users = len([u for u in self.users.values() if u.is_active])
            admin_users = len([u for u in self.users.values() if u.role == 'admin'])

            recent_logins = len([u for u in self.users.values()
                               if u.last_login and
                               datetime.fromisoformat(u.last_login) > datetime.now() - timedelta(days=7)])

            total_predictions = sum(u.total_predictions for u in self.users.values())
            total_uploads = sum(u.total_uploads for u in self.users.values())

            return {
                'total_users': total_users,
                'active_users': active_users,
                'inactive_users': total_users - active_users,
                'admin_users': admin_users,
                'recent_logins': recent_logins,
                'total_predictions': total_predictions,
                'total_uploads': total_uploads
            }
        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return {}

    def get_system_monitoring_data(self):
        """Get comprehensive system monitoring data for admin panel"""
        try:
            import os
            import psutil
            from datetime import datetime, timedelta

            # User statistics
            user_stats = self.get_user_statistics()

            # System information
            system_info = {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                'uptime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}"
            }

            # Recent activity (last 24 hours)
            recent_activity = []
            cutoff_time = datetime.now() - timedelta(hours=24)

            for user in self.users.values():
                if user.last_login:
                    try:
                        login_time = datetime.fromisoformat(user.last_login)
                        if login_time > cutoff_time:
                            recent_activity.append({
                                'type': 'login',
                                'user': user.username,
                                'timestamp': user.last_login,
                                'details': f"User {user.username} logged in"
                            })
                    except:
                        pass

            # Sort by timestamp
            recent_activity.sort(key=lambda x: x['timestamp'], reverse=True)

            # File upload tracking
            upload_stats = {
                'total_uploads': sum(u.total_uploads for u in self.users.values()),
                'total_predictions': sum(u.total_predictions for u in self.users.values()),
                'active_sessions': len([u for u in self.users.values() if u.last_login and
                                      datetime.fromisoformat(u.last_login) > datetime.now() - timedelta(hours=1)])
            }

            return {
                'user_stats': user_stats,
                'system_info': system_info,
                'recent_activity': recent_activity[:50],  # Last 50 activities
                'upload_stats': upload_stats,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting system monitoring data: {e}")
            return {
                'user_stats': self.get_user_statistics(),
                'system_info': {'error': 'System monitoring unavailable'},
                'recent_activity': [],
                'upload_stats': {'error': 'Upload stats unavailable'},
                'timestamp': datetime.now().isoformat()
            }

    def get_security_events(self, limit=100):
        """Get recent security events for admin monitoring"""
        try:
            security_events = []

            # Check for failed login attempts
            for user in self.users.values():
                if user.login_attempts > 0:
                    security_events.append({
                        'type': 'failed_login',
                        'user': user.username,
                        'attempts': user.login_attempts,
                        'locked': user.is_locked(),
                        'timestamp': user.updated_at,
                        'severity': 'high' if user.login_attempts >= 3 else 'medium'
                    })

            # Check for locked accounts
            locked_users = [u for u in self.users.values() if u.is_locked()]
            for user in locked_users:
                security_events.append({
                    'type': 'account_locked',
                    'user': user.username,
                    'locked_until': user.locked_until.isoformat() if user.locked_until else None,
                    'timestamp': user.updated_at,
                    'severity': 'high'
                })

            # Sort by timestamp
            security_events.sort(key=lambda x: x['timestamp'], reverse=True)

            return security_events[:limit]

        except Exception as e:
            logger.error(f"Error getting security events: {e}")
            return []
    
    def get_all_users(self):
        """Get all users (admin only)"""
        return list(self.users.values())

# Global user manager instance
user_manager = UserManager()

def validate_password_strength(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    
    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is strong"

def sanitize_input(input_string, max_length=1000):
    """Sanitize user input to prevent injection attacks"""
    if not input_string:
        return ""
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`', '$']
    sanitized = str(input_string)
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    # Limit length
    sanitized = sanitized[:max_length]
    
    # Strip whitespace
    sanitized = sanitized.strip()
    
    return sanitized

def log_security_event(event_type, username=None, ip_address=None, details=None):
    """Log security-related events"""
    timestamp = datetime.now().isoformat()
    log_entry = f"[SECURITY] {timestamp} - {event_type}"
    
    if username:
        log_entry += f" - User: {username}"
    if ip_address:
        log_entry += f" - IP: {ip_address}"
    if details:
        log_entry += f" - Details: {details}"
    
    logger.warning(log_entry)
    
    # Also write to security log file
    try:
        with open('security.log', 'a') as f:
            f.write(log_entry + '\n')
    except Exception as e:
        logger.error(f"Failed to write to security log: {e}")
