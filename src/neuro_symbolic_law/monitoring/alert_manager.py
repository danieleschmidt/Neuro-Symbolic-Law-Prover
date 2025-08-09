"""
Intelligent alerting and notification system for production monitoring.
"""

import time
import asyncio
import logging
import json
import smtplib
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import weakref
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"


@dataclass
class Alert:
    """Represents an alert with its metadata."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    threshold: float
    current_value: float
    condition: str
    timestamp: float
    resolved_timestamp: Optional[float] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    runbook_url: Optional[str] = None
    escalation_level: int = 0
    notification_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            severity=AlertSeverity(data['severity']),
            status=AlertStatus(data['status']),
            metric_name=data['metric_name'],
            threshold=data['threshold'],
            current_value=data['current_value'],
            condition=data['condition'],
            timestamp=data['timestamp'],
            resolved_timestamp=data.get('resolved_timestamp'),
            acknowledged_by=data.get('acknowledged_by'),
            acknowledged_at=data.get('acknowledged_at'),
            labels=data.get('labels', {}),
            runbook_url=data.get('runbook_url'),
            escalation_level=data.get('escalation_level', 0),
            notification_count=data.get('notification_count', 0)
        )
    
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == AlertStatus.ACTIVE
    
    def age(self) -> float:
        """Get alert age in seconds."""
        return time.time() - self.timestamp
    
    def format_message(self) -> str:
        """Format alert as human-readable message."""
        status_emoji = {
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.HIGH: "⚠️",
            AlertSeverity.MEDIUM: "⚡",
            AlertSeverity.LOW: "ℹ️",
            AlertSeverity.INFO: "📝"
        }
        
        emoji = status_emoji.get(self.severity, "⚠️")
        
        message = f"{emoji} **{self.name}** ({self.severity.value.upper()})\n\n"
        message += f"**Description:** {self.description}\n"
        message += f"**Metric:** {self.metric_name}\n"
        message += f"**Current Value:** {self.current_value}\n"
        message += f"**Threshold:** {self.threshold}\n"
        message += f"**Condition:** {self.condition}\n"
        message += f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(self.timestamp))}\n"
        
        if self.labels:
            labels_str = ", ".join(f"{k}={v}" for k, v in self.labels.items())
            message += f"**Labels:** {labels_str}\n"
        
        if self.runbook_url:
            message += f"**Runbook:** {self.runbook_url}\n"
        
        return message


@dataclass
class AlertRule:
    """Defines conditions for triggering alerts."""
    name: str
    metric_name: str
    condition: str  # e.g., "gt", "lt", "eq", "ne"
    threshold: float
    duration: float  # How long condition must persist
    severity: AlertSeverity
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    runbook_url: Optional[str] = None
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    cooldown_period: float = 300  # 5 minutes default cooldown
    
    def evaluate(self, current_value: float) -> bool:
        """Evaluate if current value triggers the alert condition."""
        if self.condition == "gt":
            return current_value > self.threshold
        elif self.condition == "lt":
            return current_value < self.threshold
        elif self.condition == "eq":
            return abs(current_value - self.threshold) < 1e-10
        elif self.condition == "ne":
            return abs(current_value - self.threshold) >= 1e-10
        elif self.condition == "gte":
            return current_value >= self.threshold
        elif self.condition == "lte":
            return current_value <= self.threshold
        else:
            return False


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    channel: NotificationChannel
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    rate_limit: Optional[int] = None  # Max notifications per hour
    
    # Email specific
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    recipients: List[str] = field(default_factory=list)
    
    # Webhook specific
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    
    # Slack specific
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None


class AlertManager:
    """
    Advanced alert management system with intelligent notification routing.
    
    Features:
    - Rule-based alerting with complex conditions
    - Multiple notification channels
    - Alert escalation and acknowledgment
    - Rate limiting and deduplication  
    - Alert correlation and grouping
    - Maintenance windows and suppression
    """
    
    def __init__(self,
                 storage_path: Optional[str] = None,
                 enable_escalation: bool = True,
                 default_cooldown: float = 300,
                 max_alerts: int = 10000):
        """
        Initialize alert manager.
        
        Args:
            storage_path: Path to store alert history
            enable_escalation: Enable alert escalation
            default_cooldown: Default cooldown period for alerts
            max_alerts: Maximum alerts to keep in memory
        """
        self.storage_path = storage_path or "/tmp/alerts.json"
        self.enable_escalation = enable_escalation
        self.default_cooldown = default_cooldown
        self.max_alerts = max_alerts
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.suppressed_alerts: Set[str] = set()
        
        # Notification configuration
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        
        # Alert evaluation state
        self.metric_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.last_alert_times: Dict[str, float] = {}
        
        # Rate limiting
        self.notification_counts: Dict[Tuple[str, NotificationChannel], deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Background tasks
        self.evaluation_task: Optional[asyncio.Task] = None
        self.escalation_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Alert listeners
        self.alert_listeners: List[Callable[[Alert], None]] = []
        
        # Load existing alerts
        self._load_alerts()
        
        logger.info("AlertManager initialized")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self.lock:
            self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        with self.lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
        return False
    
    def configure_notification(self, config: NotificationConfig):
        """Configure a notification channel."""
        self.notification_configs[config.channel] = config
        logger.info(f"Configured notification channel: {config.channel.value}")
    
    async def start(self):
        """Start alert processing."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        
        if self.enable_escalation:
            self.escalation_task = asyncio.create_task(self._escalation_loop())
        
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Alert manager started")
    
    async def stop(self):
        """Stop alert processing."""
        self.running = False
        
        # Cancel tasks
        for task in [self.evaluation_task, self.escalation_task, self.cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save alerts
        self._save_alerts()
        
        logger.info("Alert manager stopped")
    
    async def evaluate_metric(self, metric_name: str, value: float, 
                            labels: Optional[Dict[str, str]] = None):
        """Evaluate a metric value against alert rules."""
        current_time = time.time()
        
        with self.lock:
            # Find matching rules
            matching_rules = [
                rule for rule in self.alert_rules.values()
                if rule.metric_name == metric_name and rule.enabled
            ]
        
        for rule in matching_rules:
            await self._evaluate_rule(rule, value, current_time, labels or {})
    
    async def _evaluate_rule(self, rule: AlertRule, value: float, 
                           timestamp: float, labels: Dict[str, str]):
        """Evaluate a specific rule against a metric value."""
        rule_key = f"{rule.name}:{json.dumps(labels, sort_keys=True)}"
        
        # Check if condition is met
        condition_met = rule.evaluate(value)
        
        # Get or initialize state for this rule
        if rule_key not in self.metric_states:
            self.metric_states[rule_key] = {
                'condition_start': None,
                'condition_met': False,
                'last_value': None,
                'last_check': timestamp
            }
        
        state = self.metric_states[rule_key]
        
        if condition_met:
            # Condition is met
            if not state['condition_met']:
                # Condition just became true
                state['condition_start'] = timestamp
                state['condition_met'] = True
            
            # Check if condition has persisted long enough
            condition_duration = timestamp - state['condition_start']
            
            if condition_duration >= rule.duration:
                # Check cooldown period
                last_alert_time = self.last_alert_times.get(rule_key, 0)
                if timestamp - last_alert_time >= rule.cooldown_period:
                    await self._trigger_alert(rule, value, timestamp, labels)
        else:
            # Condition is not met
            if state['condition_met']:
                # Condition just became false - resolve any active alert
                await self._resolve_alert(rule_key)
            
            state['condition_met'] = False
            state['condition_start'] = None
        
        # Update state
        state['last_value'] = value
        state['last_check'] = timestamp
    
    async def _trigger_alert(self, rule: AlertRule, value: float, 
                           timestamp: float, labels: Dict[str, str]):
        """Trigger a new alert."""
        rule_key = f"{rule.name}:{json.dumps(labels, sort_keys=True)}"
        alert_id = f"{rule_key}_{int(timestamp)}"
        
        # Create alert
        alert = Alert(
            id=alert_id,
            name=rule.name,
            description=rule.description,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            metric_name=rule.metric_name,
            threshold=rule.threshold,
            current_value=value,
            condition=rule.condition,
            timestamp=timestamp,
            labels={**rule.labels, **labels},
            runbook_url=rule.runbook_url
        )
        
        with self.lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            self.last_alert_times[rule_key] = timestamp
        
        # Notify listeners
        await self._notify_listeners(alert)
        
        # Send notifications
        await self._send_notifications(alert, rule)
        
        logger.warning(f"Alert triggered: {alert.name} ({alert.severity.value})")
    
    async def _resolve_alert(self, rule_key: str):
        """Resolve active alerts for a rule."""
        current_time = time.time()
        
        with self.lock:
            resolved_alerts = []
            
            for alert_id, alert in self.active_alerts.items():
                alert_key = f"{alert.name}:{json.dumps(alert.labels, sort_keys=True)}"
                if alert_key == rule_key and alert.status == AlertStatus.ACTIVE:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_timestamp = current_time
                    resolved_alerts.append((alert_id, alert))
            
            # Remove resolved alerts from active list
            for alert_id, alert in resolved_alerts:
                del self.active_alerts[alert_id]
        
        # Notify about resolution
        for _, alert in resolved_alerts:
            await self._notify_listeners(alert)
            logger.info(f"Alert resolved: {alert.name}")
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for an alert."""
        if not rule.notification_channels:
            return
        
        for channel in rule.notification_channels:
            config = self.notification_configs.get(channel)
            if not config or not config.enabled:
                continue
            
            # Check rate limits
            if self._is_rate_limited(alert.id, channel, config):
                continue
            
            try:
                await self._send_notification(alert, channel, config)
                
                # Update rate limiting
                self._update_rate_limit(alert.id, channel)
                
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {e}")
    
    def _is_rate_limited(self, alert_id: str, channel: NotificationChannel,
                        config: NotificationConfig) -> bool:
        """Check if notification is rate limited."""
        if not config.rate_limit:
            return False
        
        key = (alert_id, channel)
        current_time = time.time()
        
        # Clean old entries (older than 1 hour)
        while self.notification_counts[key] and \
              current_time - self.notification_counts[key][0] > 3600:
            self.notification_counts[key].popleft()
        
        return len(self.notification_counts[key]) >= config.rate_limit
    
    def _update_rate_limit(self, alert_id: str, channel: NotificationChannel):
        """Update rate limiting counters."""
        key = (alert_id, channel)
        self.notification_counts[key].append(time.time())
    
    async def _send_notification(self, alert: Alert, channel: NotificationChannel,
                               config: NotificationConfig):
        """Send notification via specific channel."""
        if channel == NotificationChannel.EMAIL:
            await self._send_email_notification(alert, config)
        elif channel == NotificationChannel.WEBHOOK:
            await self._send_webhook_notification(alert, config)
        elif channel == NotificationChannel.SLACK:
            await self._send_slack_notification(alert, config)
        elif channel == NotificationChannel.CONSOLE:
            await self._send_console_notification(alert)
    
    async def _send_email_notification(self, alert: Alert, config: NotificationConfig):
        """Send email notification."""
        if not config.recipients or not config.smtp_server:
            return
        
        subject = f"[{alert.severity.value.upper()}] {alert.name}"
        body = alert.format_message()
        
        def send_email():
            try:
                msg = MIMEMultipart()
                msg['From'] = config.username
                msg['Subject'] = subject
                
                msg.attach(MIMEText(body, 'plain'))
                
                server = smtplib.SMTP(config.smtp_server, config.smtp_port)
                server.starttls()
                
                if config.username and config.password:
                    server.login(config.username, config.password)
                
                for recipient in config.recipients:
                    msg['To'] = recipient
                    server.send_message(msg)
                    del msg['To']
                
                server.quit()
                
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
        
        # Send email in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_email)
    
    async def _send_webhook_notification(self, alert: Alert, config: NotificationConfig):
        """Send webhook notification."""
        if not config.webhook_url:
            return
        
        try:
            import aiohttp
            
            payload = {
                'alert': alert.to_dict(),
                'timestamp': time.time()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.webhook_url,
                    json=payload,
                    headers=config.webhook_headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Webhook returned status {response.status}")
        
        except ImportError:
            logger.error("aiohttp not available for webhook notifications")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
    
    async def _send_slack_notification(self, alert: Alert, config: NotificationConfig):
        """Send Slack notification."""
        if not config.slack_webhook_url:
            return
        
        try:
            import aiohttp
            
            color_map = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.HIGH: "warning",
                AlertSeverity.MEDIUM: "#ff9900",
                AlertSeverity.LOW: "good",
                AlertSeverity.INFO: "#0099ff"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": f"{alert.name} ({alert.severity.value.upper()})",
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Metric",
                            "value": alert.metric_name,
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": str(alert.current_value),
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": str(alert.threshold),
                            "short": True
                        },
                        {
                            "title": "Condition",
                            "value": alert.condition,
                            "short": True
                        }
                    ],
                    "timestamp": int(alert.timestamp)
                }]
            }
            
            if config.slack_channel:
                payload["channel"] = config.slack_channel
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        logger.error(f"Slack webhook returned status {response.status}")
        
        except ImportError:
            logger.error("aiohttp not available for Slack notifications")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    async def _send_console_notification(self, alert: Alert):
        """Send console notification."""
        message = alert.format_message()
        print(f"\n{'='*50}")
        print("ALERT NOTIFICATION")
        print('='*50)
        print(message)
        print('='*50)
    
    async def _notify_listeners(self, alert: Alert):
        """Notify registered alert listeners."""
        for listener in self.alert_listeners:
            try:
                await asyncio.get_event_loop().run_in_executor(None, listener, alert)
            except Exception as e:
                logger.error(f"Error notifying alert listener: {e}")
    
    async def _evaluation_loop(self):
        """Main evaluation loop."""
        while self.running:
            try:
                await asyncio.sleep(10)  # Evaluate every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
    
    async def _escalation_loop(self):
        """Handle alert escalation."""
        while self.running:
            try:
                await self._process_escalations()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
    
    async def _process_escalations(self):
        """Process alert escalations."""
        current_time = time.time()
        
        with self.lock:
            for alert in self.active_alerts.values():
                if alert.status != AlertStatus.ACTIVE:
                    continue
                
                rule = self.alert_rules.get(alert.name)
                if not rule or not rule.escalation_rules:
                    continue
                
                alert_age = alert.age()
                
                # Check if escalation is needed
                for escalation in rule.escalation_rules:
                    escalation_time = escalation.get('after_minutes', 30) * 60
                    escalation_level = escalation.get('level', 1)
                    
                    if (alert_age >= escalation_time and 
                        alert.escalation_level < escalation_level):
                        
                        await self._escalate_alert(alert, escalation)
    
    async def _escalate_alert(self, alert: Alert, escalation: Dict[str, Any]):
        """Escalate an alert to higher severity or different channels."""
        alert.escalation_level = escalation.get('level', alert.escalation_level + 1)
        
        # Increase severity if specified
        new_severity = escalation.get('severity')
        if new_severity:
            alert.severity = AlertSeverity(new_severity)
        
        # Send notifications to escalation channels
        escalation_channels = escalation.get('channels', [])
        for channel_name in escalation_channels:
            try:
                channel = NotificationChannel(channel_name)
                config = self.notification_configs.get(channel)
                if config:
                    await self._send_notification(alert, channel, config)
            except ValueError:
                logger.error(f"Unknown notification channel: {channel_name}")
        
        logger.warning(f"Alert escalated: {alert.name} (level {alert.escalation_level})")
    
    async def _cleanup_loop(self):
        """Clean up old alerts and maintain storage limits."""
        while self.running:
            try:
                await self._cleanup_old_alerts()
                await asyncio.sleep(3600)  # Clean up every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        current_time = time.time()
        retention_period = 7 * 24 * 3600  # 7 days
        
        with self.lock:
            # Clean up alert history
            self.alert_history = [
                alert for alert in self.alert_history
                if (current_time - alert.timestamp) < retention_period
            ]
            
            # Limit history size
            if len(self.alert_history) > self.max_alerts:
                self.alert_history = self.alert_history[-self.max_alerts:]
            
            logger.debug(f"Cleaned up alert history, {len(self.alert_history)} alerts remaining")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = time.time()
                
                logger.info(f"Alert acknowledged: {alert.name} by {acknowledged_by}")
                return True
        return False
    
    def suppress_alert(self, rule_name: str, duration_seconds: float = 3600):
        """Suppress alerts for a rule temporarily."""
        self.suppressed_alerts.add(rule_name)
        
        # Schedule automatic unsuppression
        async def unsuppress_later():
            await asyncio.sleep(duration_seconds)
            self.suppressed_alerts.discard(rule_name)
            logger.info(f"Alert suppression lifted: {rule_name}")
        
        asyncio.create_task(unsuppress_later())
        logger.info(f"Alert suppressed: {rule_name} for {duration_seconds} seconds")
    
    def unsuppress_alert(self, rule_name: str):
        """Remove suppression for an alert rule."""
        self.suppressed_alerts.discard(rule_name)
        logger.info(f"Alert suppression removed: {rule_name}")
    
    def add_listener(self, listener: Callable[[Alert], None]):
        """Add alert listener."""
        self.alert_listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[Alert], None]):
        """Remove alert listener."""
        if listener in self.alert_listeners:
            self.alert_listeners.remove(listener)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history."""
        with self.lock:
            history = self.alert_history
            if limit:
                history = history[-limit:]
            return history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        with self.lock:
            active_by_severity = defaultdict(int)
            for alert in self.active_alerts.values():
                active_by_severity[alert.severity.value] += 1
            
            return {
                'running': self.running,
                'total_rules': len(self.alert_rules),
                'active_alerts': len(self.active_alerts),
                'suppressed_rules': len(self.suppressed_alerts),
                'total_history': len(self.alert_history),
                'active_by_severity': dict(active_by_severity),
                'notification_channels': len(self.notification_configs),
                'listeners': len(self.alert_listeners)
            }
    
    def _save_alerts(self):
        """Save alerts to storage."""
        try:
            data = {
                'active_alerts': [alert.to_dict() for alert in self.active_alerts.values()],
                'alert_history': [alert.to_dict() for alert in self.alert_history[-1000:]],  # Last 1000
                'suppressed_alerts': list(self.suppressed_alerts),
                'last_alert_times': self.last_alert_times,
                'timestamp': time.time()
            }
            
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")
    
    def _load_alerts(self):
        """Load alerts from storage."""
        try:
            if not Path(self.storage_path).exists():
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load active alerts
            for alert_data in data.get('active_alerts', []):
                alert = Alert.from_dict(alert_data)
                self.active_alerts[alert.id] = alert
            
            # Load alert history
            for alert_data in data.get('alert_history', []):
                alert = Alert.from_dict(alert_data)
                self.alert_history.append(alert)
            
            # Load suppressed alerts
            self.suppressed_alerts = set(data.get('suppressed_alerts', []))
            
            # Load last alert times
            self.last_alert_times = data.get('last_alert_times', {})
            
            logger.info(f"Loaded {len(self.active_alerts)} active alerts and "
                       f"{len(self.alert_history)} historical alerts")
            
        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()