from src.segmentation.mask_evaluation import *
from src.utils.configuration import Configuration
from src.utils.metautils import leq, geq

class MaskFeaturing:
    mapping = {
        'min_roundness': ('roundness',compute_roundness,geq,'min_roundness'),
        'max_eccentricity': ('eccentricity',compute_eccentricity,leq,'max_eccentricity'),
        'min_meters': ('meters',compute_meters,geq,'min_meters'),
        'max_meters': ('meters',compute_meters,leq,'max_meters'),
        'min_percentage': ('percentage',compute_percentage,geq,'min_percentage'),
        'max_percentage': ('percentage',compute_percentage,leq,'max_percentage'),
        'min_pixels': ('pixels',compute_pixels,geq,'min_pixels'),
        'max_pixels': ('pixels',compute_pixels,leq,'max_pixels')
    }

    def __init__(self):
        self.configuration = Configuration()

    def single_filter(self, mask, filter) -> bool:
        property, _, op, argument = MaskFeaturing.mapping[filter]
        argument = self.configuration.get(argument)
        property_value = mask[property]
        retval = op(property_value, argument)
        return retval

    def filter(self, mask) -> bool:
        filters = self.configuration.get('filters')
        outcomes = list(map(lambda x: self.single_filter(mask, x), filters))
        retval = all(outcomes)
        return retval

    def single_evaluation(self, mask: dict, filter: str) -> float:
        property, funct, _, _ = MaskFeaturing.mapping[filter]
        value = funct(mask)
        return property, value

    def evaluation(self, mask: dict) -> dict:
        retval = dict()
        filters = self.configuration.get('filters')
        for filter in filters:
            property, value = self.single_evaluation(mask, filter)
            retval[property] = value
        return retval
