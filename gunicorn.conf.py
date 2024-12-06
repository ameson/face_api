import multiprocessing

# 工作进程数 - 设置为CPU核心数的2-4倍
workers = multiprocessing.cpu_count() * 2 + 1

# 工作模式 - 使用gevent处理异步请求
worker_class = 'gevent'

# 超时设置
timeout = 600  # 增加超时时间到600秒

# 限制最大请求数 - 降低内存泄漏风险
max_requests = 500
max_requests_jitter = 50

# 日志级别
loglevel = 'info'

# 预加载应用
preload_app = True

# 限制工作进程的最大内存使用
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# 优雅的重启时间
graceful_timeout = 120

# 保持连接超时
keepalive = 5

# 工作进程启动超时
worker_tmp_timeout = 120

# 添加内存监控
worker_memory_limit = 512 * 1024 * 1024  # 512MB per worker
