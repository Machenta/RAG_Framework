"""
Secure Configuration API Management

This module provides secure APIs for updating configuration with proper
authentication, authorization, and audit logging.

SECURITY CONSIDERATIONS:
1. Never expose secrets through APIs
2. Implement proper authentication and authorization
3. Log all configuration changes
4. Validate all inputs
5. Use encrypted channels (HTTPS)
6. Implement rate limiting
7. Separate read/write permissions
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

class ConfigChangeType(str, Enum):
    """Types of configuration changes."""
    CREATE = "create"
    UPDATE = "update" 
    DELETE = "delete"
    RESTORE = "restore"

class ConfigAuditLog(BaseModel):
    """Audit log entry for configuration changes."""
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: str
    change_type: ConfigChangeType
    section: str
    field_path: str
    old_value: Optional[str] = Field(default=None, repr=False)  # Don't log sensitive values
    new_value: Optional[str] = Field(default=None, repr=False)  # Don't log sensitive values
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None

class SecureConfigAPI:
    """
    Secure API for configuration management with proper security controls.
    
    SECURITY FEATURES:
    - Secrets are NEVER exposed through this API
    - All changes are audited
    - Authentication required for all operations
    - Input validation and sanitization
    - Rollback capability
    """
    
    # Fields that should NEVER be exposed or modified via API
    PROTECTED_FIELDS = {
        "api_key", "key", "secret", "password", "token", 
        "connection_string", "credential", "cert", "private"
    }
    
    # Sections that require special permissions
    RESTRICTED_SECTIONS = {
        "secrets_mapping", "form_recognizer", "ai_search"
    }
    
    def __init__(self, config_instance, audit_logger: Optional[logging.Logger] = None):
        self.config = config_instance
        self.audit_logger = audit_logger or logging.getLogger("config_audit")
        self._audit_log: List[ConfigAuditLog] = []
    
    def _log_audit(self, audit_entry: ConfigAuditLog):
        """Log configuration change for audit purposes."""
        self._audit_log.append(audit_entry)
        
        # Log to external audit system (exclude sensitive values)
        log_data = {
            "timestamp": audit_entry.timestamp.isoformat(),
            "user_id": audit_entry.user_id,
            "change_type": audit_entry.change_type,
            "section": audit_entry.section,
            "field_path": audit_entry.field_path,
            "success": audit_entry.success,
            "ip_address": audit_entry.ip_address
        }
        
        if audit_entry.error_message:
            log_data["error"] = audit_entry.error_message
            
        self.audit_logger.info(f"Config change: {log_data}")
    
    def _is_field_protected(self, field_path: str) -> bool:
        """Check if a field contains sensitive information."""
        field_lower = field_path.lower()
        return any(protected in field_lower for protected in self.PROTECTED_FIELDS)
    
    def _validate_field_path(self, field_path: str) -> bool:
        """Validate that field path is safe and allowed."""
        if self._is_field_protected(field_path):
            return False
        
        # Add more validation rules as needed
        if ".." in field_path or field_path.startswith("/"):
            return False
            
        return True
    
    def get_safe_config(self, user_id: str, **auth_context) -> Dict[str, Any]:
        """
        Get configuration with all sensitive data masked.
        
        Args:
            user_id: Authenticated user identifier
            **auth_context: Additional authentication context
            
        Returns:
            Configuration dict with sensitive fields masked
        """
        try:
            # This uses the existing masking functionality
            safe_config = self.config.to_dict()
            
            self._log_audit(ConfigAuditLog(
                user_id=user_id,
                change_type=ConfigChangeType.UPDATE,  # READ operation
                section="all",
                field_path="read_config",
                success=True,
                **auth_context
            ))
            
            return safe_config
            
        except Exception as e:
            self._log_audit(ConfigAuditLog(
                user_id=user_id,
                change_type=ConfigChangeType.UPDATE,
                section="all", 
                field_path="read_config",
                success=False,
                error_message=str(e),
                **auth_context
            ))
            raise
    
    def update_non_sensitive_field(
        self, 
        user_id: str, 
        section: str, 
        field_path: str, 
        new_value: Any,
        **auth_context
    ) -> bool:
        """
        Safely update a non-sensitive configuration field.
        
        Args:
            user_id: Authenticated user identifier
            section: Configuration section (e.g., 'llm', 'experiments')
            field_path: Dot-separated path to field (e.g., 'params.max_tokens')
            new_value: New value to set
            **auth_context: Additional authentication context (ip_address, etc.)
            
        Returns:
            True if update was successful
            
        Raises:
            ValueError: If field is protected or invalid
            PermissionError: If user lacks permissions for section
        """
        full_path = f"{section}.{field_path}"
        
        # Security validations
        if not self._validate_field_path(full_path):
            raise ValueError(f"Field '{full_path}' is protected or invalid")
        
        if section in self.RESTRICTED_SECTIONS:
            raise PermissionError(f"Section '{section}' requires special permissions")
        
        # Get old value for audit (but don't log if sensitive)
        old_value = None
        try:
            old_value = str(getattr(getattr(self.config.app, section, None), field_path.split('.')[0], None))
            if self._is_field_protected(field_path):
                old_value = "<masked>"
        except:
            old_value = "<not_found>"
        
        try:
            # Perform the update using existing config mechanisms
            path_parts = ["app", section] + field_path.split('.')
            from rag_shared.utils.config import set_nested_attr
            set_nested_attr(self.config, path_parts, new_value)
            
            # Log successful change
            self._log_audit(ConfigAuditLog(
                user_id=user_id,
                change_type=ConfigChangeType.UPDATE,
                section=section,
                field_path=field_path,
                old_value=old_value if not self._is_field_protected(field_path) else "<masked>",
                new_value=str(new_value) if not self._is_field_protected(field_path) else "<masked>",
                success=True,
                **auth_context
            ))
            
            return True
            
        except Exception as e:
            # Log failed change
            self._log_audit(ConfigAuditLog(
                user_id=user_id,
                change_type=ConfigChangeType.UPDATE,
                section=section,
                field_path=field_path,
                success=False,
                error_message=str(e),
                **auth_context
            ))
            raise
    
    def get_audit_log(self, user_id: str, **auth_context) -> List[Dict[str, Any]]:
        """
        Get audit log for configuration changes.
        
        Args:
            user_id: Authenticated user identifier
            **auth_context: Additional authentication context
            
        Returns:
            List of audit log entries (with sensitive data masked)
        """
        # Convert to dict format for API response, masking sensitive data
        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "user_id": entry.user_id,
                "change_type": entry.change_type,
                "section": entry.section,
                "field_path": entry.field_path,
                "success": entry.success,
                "error_message": entry.error_message,
                # Note: old_value and new_value are excluded for security
            }
            for entry in self._audit_log
        ]
    
    def backup_config(self, user_id: str, **auth_context) -> str:
        """
        Create a backup of current configuration.
        
        Returns:
            Backup identifier for restoration
        """
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
        
        try:
            # Save current config to backup location
            # Implementation would depend on your backup strategy
            backup_success = self.config.save()  # or save to backup location
            
            self._log_audit(ConfigAuditLog(
                user_id=user_id,
                change_type=ConfigChangeType.CREATE,
                section="system",
                field_path="backup",
                new_value=backup_id,
                success=backup_success,
                **auth_context
            ))
            
            return backup_id
            
        except Exception as e:
            self._log_audit(ConfigAuditLog(
                user_id=user_id,
                change_type=ConfigChangeType.CREATE,
                section="system",
                field_path="backup",
                success=False,
                error_message=str(e),
                **auth_context
            ))
            raise

# Example usage with proper authentication
class ConfigurationAPI:
    """
    Example REST API implementation with proper security.
    
    IMPLEMENTATION NOTES:
    - Use HTTPS only
    - Implement rate limiting
    - Add API key or OAuth authentication
    - Validate all inputs
    - Log all operations
    """
    
    def __init__(self, config_instance):
        self.secure_api = SecureConfigAPI(config_instance)
    
    def authenticate_user(self, request) -> str:
        """
        Authenticate user from request.
        
        Returns:
            User ID if authenticated
            
        Raises:
            PermissionError: If authentication fails
        """
        # Implementation would check API keys, JWT tokens, etc.
        # This is a placeholder
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise PermissionError("API key required")
        
        # Validate API key (implementation specific)
        # Return user ID associated with the key
        return "authenticated_user_id"
    
    def update_experiment_setting(self, request):
        """
        Example API endpoint for updating experiment settings.
        
        POST /api/config/experiments/prompt_optimization/enabled
        {
            "value": true
        }
        """
        try:
            user_id = self.authenticate_user(request)
            
            # Extract parameters from request
            experiment_name = request.path_params["experiment_name"]
            setting_name = request.path_params["setting_name"]
            new_value = request.json()["value"]
            
            # Security: Validate experiment name and setting
            if setting_name not in ["enabled", "traffic_split", "status"]:
                raise ValueError(f"Setting '{setting_name}' is not allowed via API")
            
            # Use secure API to update
            field_path = f"experiments.{experiment_name}.{setting_name}"
            success = self.secure_api.update_non_sensitive_field(
                user_id=user_id,
                section="experiments",
                field_path=f"experiments.{experiment_name}.{setting_name}",
                new_value=new_value,
                ip_address=request.client.host,
                user_agent=request.headers.get("User-Agent")
            )
            
            return {"success": success, "message": "Configuration updated"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}, 400

# Usage example:
if __name__ == "__main__":
    from rag_shared.utils.config import Config
    
    # Initialize config
    cfg = Config(
        key_vault_name="RecoveredSpacesKV",
        config_folder="resources/configs",
        config_filename="handbook_config.yml"
    )
    
    # Create secure API
    secure_api = SecureConfigAPI(cfg)
    
    # Example: Safely update experiment setting
    try:
        success = secure_api.update_non_sensitive_field(
            user_id="admin_user",
            section="experiments",
            field_path="experiments.prompt_optimization.enabled",
            new_value=True,
            ip_address="192.168.1.100"
        )
        print(f"Update successful: {success}")
        
        # Get audit log
        audit_log = secure_api.get_audit_log("admin_user")
        print(f"Audit entries: {len(audit_log)}")
        
    except Exception as e:
        print(f"Update failed: {e}")
