"""
Real-time monitoring dashboard with customizable widgets and visualizations.
"""

import time
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


class WidgetType(Enum):
    """Types of dashboard widgets."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    GAUGE = "gauge"
    COUNTER = "counter"
    TABLE = "table"
    HEATMAP = "heatmap"
    PIE_CHART = "pie_chart"
    STATUS_INDICATOR = "status_indicator"
    TEXT_DISPLAY = "text_display"
    ALERT_LIST = "alert_list"


class RefreshRate(Enum):
    """Dashboard refresh rates."""
    REAL_TIME = 1
    FAST = 5
    MEDIUM = 15
    SLOW = 60
    VERY_SLOW = 300


@dataclass
class DataPoint:
    """Represents a single data point for visualization."""
    timestamp: float
    value: Union[int, float, str, Dict[str, Any]]
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSeries:
    """Represents a data series for charts."""
    name: str
    data: List[DataPoint] = field(default_factory=list)
    color: Optional[str] = None
    unit: Optional[str] = None
    max_points: int = 1000
    
    def add_point(self, timestamp: float, value: Union[int, float], 
                  label: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a data point to the series."""
        point = DataPoint(
            timestamp=timestamp,
            value=value,
            label=label,
            metadata=metadata or {}
        )
        
        self.data.append(point)
        
        # Maintain maximum points
        if len(self.data) > self.max_points:
            self.data = self.data[-self.max_points:]
    
    def get_recent_data(self, duration_seconds: float) -> List[DataPoint]:
        """Get data points from the last N seconds."""
        cutoff_time = time.time() - duration_seconds
        return [point for point in self.data if point.timestamp >= cutoff_time]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of the data series."""
        if not self.data:
            return {}
        
        numeric_values = [
            point.value for point in self.data 
            if isinstance(point.value, (int, float))
        ]
        
        if not numeric_values:
            return {}
        
        return {
            'count': len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'mean': statistics.mean(numeric_values),
            'median': statistics.median(numeric_values),
            'std_dev': statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
        }


@dataclass
class DashboardWidget:
    """Configuration for a dashboard widget."""
    id: str
    title: str
    widget_type: WidgetType
    data_source: str
    position: Tuple[int, int]  # (row, column)
    size: Tuple[int, int]  # (width, height)
    refresh_rate: RefreshRate = RefreshRate.MEDIUM
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    # Chart-specific configuration
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    show_legend: bool = True
    color_scheme: Optional[str] = None
    
    # Gauge-specific configuration
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Table-specific configuration
    columns: List[str] = field(default_factory=list)
    max_rows: int = 100
    
    # Filtering and aggregation
    time_window: Optional[int] = None  # seconds
    aggregation_function: Optional[str] = None  # 'avg', 'sum', 'max', 'min'
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert widget to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DashboardWidget':
        """Create widget from dictionary."""
        return cls(
            id=data['id'],
            title=data['title'],
            widget_type=WidgetType(data['widget_type']),
            data_source=data['data_source'],
            position=tuple(data['position']),
            size=tuple(data['size']),
            refresh_rate=RefreshRate(data.get('refresh_rate', RefreshRate.MEDIUM.value)),
            config=data.get('config', {}),
            enabled=data.get('enabled', True),
            x_axis_label=data.get('x_axis_label'),
            y_axis_label=data.get('y_axis_label'),
            show_legend=data.get('show_legend', True),
            color_scheme=data.get('color_scheme'),
            min_value=data.get('min_value'),
            max_value=data.get('max_value'),
            warning_threshold=data.get('warning_threshold'),
            critical_threshold=data.get('critical_threshold'),
            columns=data.get('columns', []),
            max_rows=data.get('max_rows', 100),
            time_window=data.get('time_window'),
            aggregation_function=data.get('aggregation_function'),
            filters=data.get('filters', {})
        )


class DataSource:
    """Base class for dashboard data sources."""
    
    def __init__(self, name: str):
        self.name = name
        self.series: Dict[str, DataSeries] = {}
        self.lock = threading.RLock()
    
    def add_series(self, series_name: str, series: DataSeries):
        """Add a data series to this source."""
        with self.lock:
            self.series[series_name] = series
    
    def get_series(self, series_name: str) -> Optional[DataSeries]:
        """Get a data series by name."""
        with self.lock:
            return self.series.get(series_name)
    
    def get_data(self, series_name: str, time_window: Optional[int] = None) -> List[DataPoint]:
        """Get data from a series."""
        with self.lock:
            series = self.series.get(series_name)
            if not series:
                return []
            
            if time_window:
                return series.get_recent_data(time_window)
            else:
                return series.data.copy()
    
    async def update_data(self, series_name: str, timestamp: float, value: Any,
                         label: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Update data in a series."""
        with self.lock:
            if series_name not in self.series:
                self.series[series_name] = DataSeries(name=series_name)
            
            self.series[series_name].add_point(timestamp, value, label, metadata)


class Dashboard:
    """
    Real-time monitoring dashboard system.
    
    Features:
    - Multiple customizable widgets
    - Real-time data updates
    - Multiple data sources
    - Responsive layouts
    - Export capabilities
    - User authentication and permissions
    """
    
    def __init__(self,
                 name: str,
                 storage_path: Optional[str] = None,
                 auto_save: bool = True,
                 max_data_points: int = 10000):
        """
        Initialize dashboard.
        
        Args:
            name: Dashboard name
            storage_path: Path to store dashboard configuration
            auto_save: Whether to automatically save configuration changes
            max_data_points: Maximum data points to keep per series
        """
        self.name = name
        self.storage_path = storage_path or f"/tmp/dashboard_{name}.json"
        self.auto_save = auto_save
        self.max_data_points = max_data_points
        
        # Dashboard components
        self.widgets: Dict[str, DashboardWidget] = {}
        self.data_sources: Dict[str, DataSource] = {}
        
        # Update tracking
        self.last_update: Dict[str, float] = {}
        self.update_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Background tasks
        self.refresh_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load existing configuration
        self._load_configuration()
        
        logger.info(f"Dashboard '{name}' initialized with {len(self.widgets)} widgets")
    
    def add_widget(self, widget: DashboardWidget):
        """Add a widget to the dashboard."""
        with self.lock:
            self.widgets[widget.id] = widget
            self.last_update[widget.id] = 0
        
        if self.auto_save:
            self._save_configuration()
        
        logger.info(f"Added widget: {widget.id} ({widget.widget_type.value})")
    
    def remove_widget(self, widget_id: str) -> bool:
        """Remove a widget from the dashboard."""
        with self.lock:
            if widget_id in self.widgets:
                del self.widgets[widget_id]
                self.last_update.pop(widget_id, None)
                
                if self.auto_save:
                    self._save_configuration()
                
                logger.info(f"Removed widget: {widget_id}")
                return True
        return False
    
    def update_widget(self, widget_id: str, **kwargs):
        """Update widget configuration."""
        with self.lock:
            if widget_id in self.widgets:
                widget = self.widgets[widget_id]
                
                for key, value in kwargs.items():
                    if hasattr(widget, key):
                        setattr(widget, key, value)
                
                if self.auto_save:
                    self._save_configuration()
                
                logger.debug(f"Updated widget: {widget_id}")
                return True
        return False
    
    def add_data_source(self, data_source: DataSource):
        """Add a data source to the dashboard."""
        with self.lock:
            self.data_sources[data_source.name] = data_source
        
        logger.info(f"Added data source: {data_source.name}")
    
    def get_data_source(self, name: str) -> Optional[DataSource]:
        """Get a data source by name."""
        with self.lock:
            return self.data_sources.get(name)
    
    async def start(self):
        """Start dashboard updates."""
        if self.running:
            return
        
        self.running = True
        self.refresh_task = asyncio.create_task(self._refresh_loop())
        
        logger.info(f"Dashboard '{self.name}' started")
    
    async def stop(self):
        """Stop dashboard updates."""
        self.running = False
        
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass
        
        if self.auto_save:
            self._save_configuration()
        
        logger.info(f"Dashboard '{self.name}' stopped")
    
    async def _refresh_loop(self):
        """Main refresh loop for updating widgets."""
        while self.running:
            try:
                await self._refresh_widgets()
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard refresh loop: {e}")
                await asyncio.sleep(5)
    
    async def _refresh_widgets(self):
        """Refresh widgets that need updating."""
        current_time = time.time()
        
        with self.lock:
            widgets_to_update = []
            
            for widget_id, widget in self.widgets.items():
                if not widget.enabled:
                    continue
                
                last_update = self.last_update.get(widget_id, 0)
                refresh_interval = widget.refresh_rate.value
                
                if current_time - last_update >= refresh_interval:
                    widgets_to_update.append((widget_id, widget))
        
        # Update widgets outside of lock to avoid blocking
        for widget_id, widget in widgets_to_update:
            try:
                await self._update_widget_data(widget_id, widget)
                self.last_update[widget_id] = current_time
            except Exception as e:
                logger.error(f"Error updating widget {widget_id}: {e}")
    
    async def _update_widget_data(self, widget_id: str, widget: DashboardWidget):
        """Update data for a specific widget."""
        data_source = self.data_sources.get(widget.data_source)
        if not data_source:
            logger.warning(f"Data source not found for widget {widget_id}: {widget.data_source}")
            return
        
        # Get data based on widget configuration
        data = await self._get_widget_data(widget, data_source)
        
        # Notify callbacks
        for callback in self.update_callbacks.get(widget_id, []):
            try:
                await asyncio.get_event_loop().run_in_executor(None, callback, widget, data)
            except Exception as e:
                logger.error(f"Error in widget update callback: {e}")
    
    async def _get_widget_data(self, widget: DashboardWidget, data_source: DataSource) -> Dict[str, Any]:
        """Get formatted data for a widget."""
        if widget.widget_type == WidgetType.LINE_CHART:
            return await self._get_line_chart_data(widget, data_source)
        elif widget.widget_type == WidgetType.BAR_CHART:
            return await self._get_bar_chart_data(widget, data_source)
        elif widget.widget_type == WidgetType.GAUGE:
            return await self._get_gauge_data(widget, data_source)
        elif widget.widget_type == WidgetType.COUNTER:
            return await self._get_counter_data(widget, data_source)
        elif widget.widget_type == WidgetType.TABLE:
            return await self._get_table_data(widget, data_source)
        elif widget.widget_type == WidgetType.STATUS_INDICATOR:
            return await self._get_status_data(widget, data_source)
        else:
            return await self._get_generic_data(widget, data_source)
    
    async def _get_line_chart_data(self, widget: DashboardWidget, data_source: DataSource) -> Dict[str, Any]:
        """Get data for line chart widget."""
        # Get all series from the data source
        series_data = []
        
        with data_source.lock:
            for series_name, series in data_source.series.items():
                # Apply time window filter
                if widget.time_window:
                    data_points = series.get_recent_data(widget.time_window)
                else:
                    data_points = series.data
                
                # Apply aggregation if specified
                if widget.aggregation_function and data_points:
                    data_points = self._aggregate_data_points(data_points, widget.aggregation_function)
                
                series_data.append({
                    'name': series_name,
                    'data': [{'x': p.timestamp * 1000, 'y': p.value} for p in data_points if isinstance(p.value, (int, float))],
                    'color': series.color,
                    'unit': series.unit
                })
        
        return {
            'type': 'line_chart',
            'series': series_data,
            'x_axis_label': widget.x_axis_label or 'Time',
            'y_axis_label': widget.y_axis_label or 'Value',
            'show_legend': widget.show_legend,
            'color_scheme': widget.color_scheme
        }
    
    async def _get_bar_chart_data(self, widget: DashboardWidget, data_source: DataSource) -> Dict[str, Any]:
        """Get data for bar chart widget."""
        categories = []
        series_data = []
        
        with data_source.lock:
            for series_name, series in data_source.series.items():
                if widget.time_window:
                    data_points = series.get_recent_data(widget.time_window)
                else:
                    data_points = series.data[-100:]  # Last 100 points
                
                if data_points:
                    # Group by label if available, otherwise use recent values
                    if data_points[0].label:
                        grouped = defaultdict(list)
                        for point in data_points:
                            if point.label:
                                grouped[point.label].append(point.value)
                        
                        categories = list(grouped.keys())
                        values = [statistics.mean(vals) if vals else 0 for vals in grouped.values()]
                    else:
                        categories = [f"Point {i}" for i in range(len(data_points[-10:]))]
                        values = [p.value for p in data_points[-10:] if isinstance(p.value, (int, float))]
                    
                    series_data.append({
                        'name': series_name,
                        'data': values
                    })
        
        return {
            'type': 'bar_chart',
            'categories': categories,
            'series': series_data,
            'x_axis_label': widget.x_axis_label or 'Category',
            'y_axis_label': widget.y_axis_label or 'Value'
        }
    
    async def _get_gauge_data(self, widget: DashboardWidget, data_source: DataSource) -> Dict[str, Any]:
        """Get data for gauge widget."""
        # Get the most recent value from the first series
        current_value = 0
        series_name = None
        
        with data_source.lock:
            for name, series in data_source.series.items():
                if series.data:
                    latest_point = series.data[-1]
                    if isinstance(latest_point.value, (int, float)):
                        current_value = latest_point.value
                        series_name = name
                        break
        
        return {
            'type': 'gauge',
            'value': current_value,
            'min_value': widget.min_value or 0,
            'max_value': widget.max_value or 100,
            'warning_threshold': widget.warning_threshold,
            'critical_threshold': widget.critical_threshold,
            'series_name': series_name
        }
    
    async def _get_counter_data(self, widget: DashboardWidget, data_source: DataSource) -> Dict[str, Any]:
        """Get data for counter widget."""
        counters = {}
        
        with data_source.lock:
            for series_name, series in data_source.series.items():
                if series.data:
                    # Get the latest value or sum of recent values
                    if widget.aggregation_function == 'sum' and widget.time_window:
                        recent_points = series.get_recent_data(widget.time_window)
                        value = sum(p.value for p in recent_points if isinstance(p.value, (int, float)))
                    else:
                        latest_point = series.data[-1]
                        value = latest_point.value if isinstance(latest_point.value, (int, float)) else 0
                    
                    counters[series_name] = {
                        'value': value,
                        'unit': series.unit,
                        'change': self._calculate_change(series)
                    }
        
        return {
            'type': 'counter',
            'counters': counters
        }
    
    async def _get_table_data(self, widget: DashboardWidget, data_source: DataSource) -> Dict[str, Any]:
        """Get data for table widget."""
        rows = []
        columns = widget.columns or ['Timestamp', 'Series', 'Value']
        
        # Collect recent data from all series
        all_points = []
        
        with data_source.lock:
            for series_name, series in data_source.series.items():
                recent_data = series.data[-widget.max_rows:] if series.data else []
                for point in recent_data:
                    all_points.append({
                        'timestamp': point.timestamp,
                        'series': series_name,
                        'value': point.value,
                        'label': point.label,
                        **point.metadata
                    })
        
        # Sort by timestamp (newest first)
        all_points.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Convert to table rows
        for point in all_points[:widget.max_rows]:
            row = {}
            for col in columns:
                if col.lower() == 'timestamp':
                    row[col] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(point['timestamp']))
                elif col.lower() == 'series':
                    row[col] = point['series']
                elif col.lower() == 'value':
                    row[col] = point['value']
                else:
                    row[col] = point.get(col, '')
            rows.append(row)
        
        return {
            'type': 'table',
            'columns': columns,
            'rows': rows
        }
    
    async def _get_status_data(self, widget: DashboardWidget, data_source: DataSource) -> Dict[str, Any]:
        """Get data for status indicator widget."""
        status = 'unknown'
        message = 'No data available'
        
        with data_source.lock:
            for series_name, series in data_source.series.items():
                if series.data:
                    latest_point = series.data[-1]
                    
                    if isinstance(latest_point.value, str):
                        status = latest_point.value.lower()
                        message = latest_point.label or f"{series_name}: {latest_point.value}"
                    elif isinstance(latest_point.value, (int, float)):
                        # Determine status based on thresholds
                        value = latest_point.value
                        if widget.critical_threshold and value >= widget.critical_threshold:
                            status = 'critical'
                        elif widget.warning_threshold and value >= widget.warning_threshold:
                            status = 'warning'
                        else:
                            status = 'healthy'
                        
                        message = f"{series_name}: {value}"
                    break
        
        return {
            'type': 'status_indicator',
            'status': status,
            'message': message,
            'timestamp': time.time()
        }
    
    async def _get_generic_data(self, widget: DashboardWidget, data_source: DataSource) -> Dict[str, Any]:
        """Get generic data for unsupported widget types."""
        return {
            'type': widget.widget_type.value,
            'data': 'Widget type not implemented',
            'timestamp': time.time()
        }
    
    def _aggregate_data_points(self, data_points: List[DataPoint], 
                              aggregation_function: str) -> List[DataPoint]:
        """Apply aggregation function to data points."""
        if not data_points or aggregation_function not in ['avg', 'sum', 'max', 'min']:
            return data_points
        
        # Group data points by time buckets (1 minute intervals)
        bucket_size = 60  # 1 minute
        buckets = defaultdict(list)
        
        for point in data_points:
            bucket_time = int(point.timestamp / bucket_size) * bucket_size
            buckets[bucket_time].append(point)
        
        # Aggregate each bucket
        aggregated_points = []
        for bucket_time, points in sorted(buckets.items()):
            numeric_values = [p.value for p in points if isinstance(p.value, (int, float))]
            
            if numeric_values:
                if aggregation_function == 'avg':
                    aggregated_value = statistics.mean(numeric_values)
                elif aggregation_function == 'sum':
                    aggregated_value = sum(numeric_values)
                elif aggregation_function == 'max':
                    aggregated_value = max(numeric_values)
                elif aggregation_function == 'min':
                    aggregated_value = min(numeric_values)
                else:
                    aggregated_value = numeric_values[0]
                
                aggregated_points.append(DataPoint(
                    timestamp=bucket_time,
                    value=aggregated_value,
                    label=f"{aggregation_function}({len(points)} points)"
                ))
        
        return aggregated_points
    
    def _calculate_change(self, series: DataSeries) -> Optional[float]:
        """Calculate percentage change for the most recent data point."""
        if len(series.data) < 2:
            return None
        
        latest = series.data[-1]
        previous = series.data[-2]
        
        if isinstance(latest.value, (int, float)) and isinstance(previous.value, (int, float)):
            if previous.value == 0:
                return None
            return ((latest.value - previous.value) / previous.value) * 100
        
        return None
    
    def add_update_callback(self, widget_id: str, callback: Callable):
        """Add callback for widget updates."""
        self.update_callbacks[widget_id].append(callback)
    
    def remove_update_callback(self, widget_id: str, callback: Callable):
        """Remove widget update callback."""
        if widget_id in self.update_callbacks:
            try:
                self.update_callbacks[widget_id].remove(callback)
            except ValueError:
                pass
    
    def get_layout(self) -> Dict[str, Any]:
        """Get dashboard layout information."""
        with self.lock:
            return {
                'name': self.name,
                'widgets': [widget.to_dict() for widget in self.widgets.values()],
                'data_sources': list(self.data_sources.keys()),
                'last_update': max(self.last_update.values()) if self.last_update else 0
            }
    
    def export_data(self, format: str = 'json', time_window: Optional[int] = None) -> str:
        """Export dashboard data in specified format."""
        export_data = {
            'dashboard_name': self.name,
            'export_timestamp': time.time(),
            'time_window': time_window,
            'data_sources': {}
        }
        
        with self.lock:
            for source_name, data_source in self.data_sources.items():
                source_data = {}
                
                with data_source.lock:
                    for series_name, series in data_source.series.items():
                        if time_window:
                            data_points = series.get_recent_data(time_window)
                        else:
                            data_points = series.data
                        
                        source_data[series_name] = {
                            'unit': series.unit,
                            'color': series.color,
                            'statistics': series.get_statistics(),
                            'data': [
                                {
                                    'timestamp': p.timestamp,
                                    'value': p.value,
                                    'label': p.label,
                                    'metadata': p.metadata
                                }
                                for p in data_points
                            ]
                        }
                
                export_data['data_sources'][source_name] = source_data
        
        if format.lower() == 'json':
            return json.dumps(export_data, indent=2)
        else:
            # Could add CSV, XML, etc. export formats
            return json.dumps(export_data, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        with self.lock:
            total_data_points = 0
            widget_types = defaultdict(int)
            
            for widget in self.widgets.values():
                widget_types[widget.widget_type.value] += 1
            
            for data_source in self.data_sources.values():
                with data_source.lock:
                    for series in data_source.series.values():
                        total_data_points += len(series.data)
            
            return {
                'name': self.name,
                'running': self.running,
                'total_widgets': len(self.widgets),
                'enabled_widgets': sum(1 for w in self.widgets.values() if w.enabled),
                'widget_types': dict(widget_types),
                'total_data_sources': len(self.data_sources),
                'total_data_points': total_data_points,
                'auto_save': self.auto_save,
                'storage_path': self.storage_path
            }
    
    def _save_configuration(self):
        """Save dashboard configuration to file."""
        try:
            config = {
                'name': self.name,
                'widgets': [widget.to_dict() for widget in self.widgets.values()],
                'auto_save': self.auto_save,
                'max_data_points': self.max_data_points,
                'timestamp': time.time()
            }
            
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(config, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save dashboard configuration: {e}")
    
    def _load_configuration(self):
        """Load dashboard configuration from file."""
        try:
            if not Path(self.storage_path).exists():
                return
            
            with open(self.storage_path, 'r') as f:
                config = json.load(f)
            
            # Load widgets
            for widget_data in config.get('widgets', []):
                widget = DashboardWidget.from_dict(widget_data)
                self.widgets[widget.id] = widget
                self.last_update[widget.id] = 0
            
            logger.info(f"Loaded dashboard configuration with {len(self.widgets)} widgets")
            
        except Exception as e:
            logger.error(f"Failed to load dashboard configuration: {e}")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


# Pre-configured widget templates
class WidgetTemplates:
    """Collection of pre-configured widget templates."""
    
    @staticmethod
    def system_cpu_chart(position: Tuple[int, int] = (0, 0)) -> DashboardWidget:
        """CPU usage line chart widget."""
        return DashboardWidget(
            id="system_cpu_chart",
            title="CPU Usage",
            widget_type=WidgetType.LINE_CHART,
            data_source="system_metrics",
            position=position,
            size=(2, 1),
            refresh_rate=RefreshRate.FAST,
            y_axis_label="CPU %",
            time_window=3600,  # 1 hour
            max_value=100
        )
    
    @staticmethod
    def memory_gauge(position: Tuple[int, int] = (0, 2)) -> DashboardWidget:
        """Memory usage gauge widget."""
        return DashboardWidget(
            id="memory_gauge",
            title="Memory Usage",
            widget_type=WidgetType.GAUGE,
            data_source="system_metrics",
            position=position,
            size=(1, 1),
            refresh_rate=RefreshRate.MEDIUM,
            min_value=0,
            max_value=100,
            warning_threshold=80,
            critical_threshold=95
        )
    
    @staticmethod
    def alert_status(position: Tuple[int, int] = (1, 0)) -> DashboardWidget:
        """Alert status indicator widget."""
        return DashboardWidget(
            id="alert_status",
            title="Alert Status",
            widget_type=WidgetType.STATUS_INDICATOR,
            data_source="alerts",
            position=position,
            size=(1, 1),
            refresh_rate=RefreshRate.REAL_TIME
        )
    
    @staticmethod
    def request_counter(position: Tuple[int, int] = (1, 1)) -> DashboardWidget:
        """Request counter widget."""
        return DashboardWidget(
            id="request_counter",
            title="Total Requests",
            widget_type=WidgetType.COUNTER,
            data_source="application_metrics",
            position=position,
            size=(1, 1),
            refresh_rate=RefreshRate.FAST,
            aggregation_function="sum",
            time_window=300  # 5 minutes
        )