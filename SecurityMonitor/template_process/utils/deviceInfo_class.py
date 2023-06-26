class DeviceInfo:
    def __init__(self, sn, meterType, x, y, w, h):
        self.y = None
        self.sn = sn
        self.meterType = meterType
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.pointerMeter = ''
        self.lightMeter = ''

    def SetPointerMeterVal(self, val):
        self.pointerMeter = val

    def SetLightMeterVal(self, val):
        self.lightMeter = val

    def SetPosition(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
