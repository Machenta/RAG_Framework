# Configuration Security Guide

## 🚨 CRITICAL SECURITY CONSIDERATIONS

### **1. API Configuration Updates - SECURITY RISKS**

**❌ NEVER DO THIS:**
```python
# Exposing config updates directly via API
@app.post("/api/config/update")
def update_config(config_data: dict):
    cfg.app = AppConfig.model_validate(config_data)  # DANGEROUS!
    return {"status": "updated"}
```

**✅ SECURE APPROACH:**
```python
# Use the SecureConfigAPI class
from rag_shared.utils.secure_config_api import SecureConfigAPI

@app.post("/api/config/experiments/{experiment_name}/enabled")
def update_experiment_enabled(experiment_name: str, enabled: bool, user: User = Depends(get_current_user)):
    secure_api = SecureConfigAPI(cfg)
    success = secure_api.update_non_sensitive_field(
        user_id=user.id,
        section="experiments", 
        field_path=f"experiments.{experiment_name}.enabled",
        new_value=enabled,
        ip_address=request.client.host
    )
    return {"success": success}
```

### **2. Secrets Management - SECURITY BEST PRACTICES**

**❌ PROBLEMS WITH HARDCODED DEFAULTS:**
```python
# This was the problematic code:
empty_secrets = SecretsMapping(
    AzureSearchAPIKey=[],          # Hardcoded secret names
    AzureSearchEmbeddingAPIKey=[], # Inflexible
    OpenAIAPIKey=[],              # Security risk
    FormRecognizerAPIKey=[]       # Assumes specific secrets
)
```

**✅ SECURE SOLUTION:**
```python
# Now we handle missing secrets properly:
if not hasattr(self.app, "secrets_mapping") or self.app.secrets_mapping is None:
    logging.warning("AppConfig missing 'secrets_mapping' attribute; skipping secret injection")
    self.secrets_mapping = None  # Let application handle this explicitly
```

### **3. Configuration Security Layers**

#### **Layer 1: Field-Level Protection**
```python
PROTECTED_FIELDS = {
    "api_key", "key", "secret", "password", "token", 
    "connection_string", "credential", "cert", "private"
}
```

#### **Layer 2: Section-Level Restrictions** 
```python
RESTRICTED_SECTIONS = {
    "secrets_mapping", "form_recognizer", "ai_search"
}
```

#### **Layer 3: Audit Logging**
```python
# All changes are logged with:
- User ID
- Timestamp  
- Changed fields (values masked if sensitive)
- IP address
- Success/failure status
```

### **4. API Security Requirements**

For production API deployments:

1. **Authentication & Authorization**
   ```python
   @require_api_key
   @require_role("config_admin")
   def update_config_endpoint():
       pass
   ```

2. **HTTPS Only**
   ```python
   # Redirect HTTP to HTTPS
   app.add_middleware(HTTPSRedirectMiddleware)
   ```

3. **Rate Limiting**
   ```python
   @limiter.limit("10/minute")
   def update_config_endpoint():
       pass
   ```

4. **Input Validation**
   ```python
   # Use Pydantic models for all inputs
   class ExperimentUpdateRequest(BaseModel):
       enabled: bool
       traffic_split: Optional[int] = Field(ge=0, le=100)
   ```

5. **Audit Trail**
   ```python
   # Log to external audit system
   audit_logger.info({
       "user_id": user.id,
       "action": "config_update",
       "section": "experiments",
       "timestamp": datetime.now(),
       "ip": request.client.host
   })
   ```

### **5. Environment-Specific Security**

#### **Development Environment**
```yaml
# config/dev.yml
app:
  other:
    debug: true
    log_level: DEBUG
  # Minimal secrets for development
```

#### **Production Environment**
```yaml
# config/prod.yml  
app:
  other:
    debug: false
    log_level: WARNING
  # All secrets from Key Vault
  # No default values for sensitive fields
```

### **6. Secret Injection Security**

**✅ SECURE Pattern:**
```python
def _inject_secrets(self):
    if self.secrets_mapping is None:
        logging.info("No secrets mapping configured, skipping secret injection")
        return  # Fail safe - don't create defaults
    
    for secret_name, attr_path in mapping.items():
        if attr_path is None:
            continue  # Skip optional secrets
        try:
            secret_value = self._get_secret(secret_name)
            if secret_value is not None:
                set_nested_attr(self, list(attr_path), secret_value)
        except Exception as e:
            logging.error(f"Failed to inject secret '{secret_name}': {e}")
            # Continue with other secrets
```

### **7. Configuration Backup & Recovery**

```python
# Before any API changes
backup_id = secure_api.backup_config(user_id="admin")

# If something goes wrong
secure_api.restore_config(backup_id, user_id="admin")
```

### **8. Monitoring & Alerting**

```python
# Set up alerts for:
- Multiple failed secret retrievals
- Unauthorized configuration access attempts  
- Unusual configuration change patterns
- Configuration changes outside business hours
```

## 🔐 DEPLOYMENT CHECKLIST

- [ ] All secrets in Azure Key Vault (not in config files)
- [ ] API authentication implemented
- [ ] HTTPS enforced
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] Input validation in place
- [ ] Backup/restore procedures tested
- [ ] Monitoring and alerting configured
- [ ] Security testing completed
- [ ] Documentation updated

## 🚫 NEVER EXPOSE VIA API

- API keys or secrets
- Connection strings
- Authentication tokens
- Key Vault names or URIs
- Internal system details
- Debug information in production
- Stack traces to external users
- Detailed error messages

## ✅ SAFE TO EXPOSE VIA API

- Experiment enabled/disabled flags
- Traffic split percentages (0-100)
- Max tokens limits (within bounds)
- Temperature settings (within bounds)
- Non-sensitive feature flags
- Public configuration options
