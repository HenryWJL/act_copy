class Logger:
    """Information logger"""
    def __init__(self, path: str):
        self.root = open(path, 'a')

    def dump(self, info: str):
        print(info)
        self.root.write(info + '\n')
        self.root.flush()

    def close(self):
        self.root.close()