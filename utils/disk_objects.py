class DiskList:
    # stupid object that is a list of disks so that pickle can picklify a list of objects
    def __init__(self, disks):
        self.disks = disks

# Converting lat/long to cartesian
class Disk:
    def __init__(self, id, x, y, colour):
        self.id = id
        self.x = x
        self.y = y
        self.colour = colour

class DetectedDisk:
    '''
    Class to store representations of a detected disk.
    Currently these come from a basic computer vision script, but eventually will come from a machine learning model

    Future work includes identifying which cluster each disk belongs to, in order to more efficienly run
    the processing algorithm.
    '''

    def __init__(self, id, x, y, angle, colour):

        self.id    = id
        self.x     = x
        self.y     = y
        self.angle = angle
        self.colour = colour

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

class PredictedLandingZone:
    '''
    Class to store representations of a predicted landing zone, which come from two detected disks being run through the
    processing algorithm. Lat and Lon both come from the cartesian positions of the disks and could have accumulated errors due to
    the approximation we use for the conversion.
    '''

    def __init__(self, id, lat, lon, color):

        self.id    = id
        self.lat   = lat
        self.lon   = lon
        self.type = color