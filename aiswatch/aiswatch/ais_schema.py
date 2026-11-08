"""Original 26 AIS fields; MMSI/IMO remain textual identifiers, numeric measurements are doubles."""
COLUMNS = ['# Timestamp', 'Type of mobile', 'MMSI', 'Latitude', 'Longitude',
 'Navigational status', 'ROT', 'SOG', 'COG', 'Heading', 'IMO', 'Callsign', 'Name',
 'Ship type', 'Cargo type', 'Width', 'Length', 'Type of position fixing device',
 'Draught', 'Destination', 'ETA', 'Data source type', 'A', 'B', 'C', 'D']
NUMERIC = ['Latitude','Longitude','ROT','SOG','COG','Heading','Width','Length','Draught','A','B','C','D']
FLAGS = ['flag_bad_pos','flag_not_class_a','flag_bad_mmsi']

