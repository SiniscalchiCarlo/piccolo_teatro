LOG_CONFIG = {
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {
    "single-process": {
      "class": "logging.Formatter",
      "format": "%(message)s"
    },
    "multi-process": {
      "class": "logging.Formatter",
      "format": "%(message)s"

    },
    "multi-thread": {
      "class": "logging.Formatter",
      "format": " %(message)s"
    },
  },

  "handlers": {
    "console": {
      "level": "DEBUG",
      "class": "logging.StreamHandler",
      "formatter": "single-process",
      "stream": "ext://sys.stdout"
    }
  },

  "loggers": {},
  "root": {
    "handlers": ["console", ],
    "level": "INFO"
  }
}