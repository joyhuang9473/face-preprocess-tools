# -*- coding: utf-8 -*-
"""
reference: https://github.com/luoyetx/deep-landmark

Copyright (c) 2015, zhangjie
All rights reserved.
"""
import time

class Logger(object):

    def __init__(self, logfilePath = None):
        if logfilePath:
            self.logFile = open(logfilePath, 'w')

    def printMsg(msg):
        """
        log message
        """
        now = time.ctime()
        print("[%s] %s" % (now, msg))

    def writeMsg(self, msg):
        self.logFile.write(msg)
