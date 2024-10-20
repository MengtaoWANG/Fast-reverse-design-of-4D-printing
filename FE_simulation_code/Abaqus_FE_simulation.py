from part import *
from material import *
from section import *
from optimization import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import numpy as np


data_length = 10
def input_data_fu(data_length):
    input_data = np.random.randint(0, 2, (2, data_length))
    return input_data

mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
mdb.models['Model-1'].sketches['__profile__'].rectangle(point1=(-45.0, 20.0),
    point2=(-5.0, 0.0))
mdb.models['Model-1'].sketches['__profile__'].ObliqueDimension(textPoint=(
    -57.5664672851562, 7.22222518920898), value=2.5, vertex1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[0], vertex2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[1])
mdb.models['Model-1'].sketches['__profile__'].ObliqueDimension(textPoint=(
    -21.8706321716309, -6.20369911193848), value=5.0, vertex1=
    mdb.models['Model-1'].sketches['__profile__'].vertices[1], vertex2=
    mdb.models['Model-1'].sketches['__profile__'].vertices[2])
mdb.models['Model-1'].Part(dimensionality=THREE_D, name='Part-1', type=
    DEFORMABLE_BODY)
mdb.models['Model-1'].parts['Part-1'].BaseSolidExtrude(depth=10.0, sketch=
    mdb.models['Model-1'].sketches['__profile__'])
del mdb.models['Model-1'].sketches['__profile__']
mdb.models['Model-1'].Part(name='Part-2', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-3', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-4', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-5', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-6', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-7', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-8', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-9', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-10', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-11', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-12', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-13', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-14', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-15', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-16', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-17', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-18', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-19', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Part(name='Part-20', objectToCopy=
    mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].Material(name='Material-0times-30min')
mdb.models['Model-1'].materials['Material-0times-30min'].Conductivity(table=((
167000.0, ), ))
mdb.models['Model-1'].materials['Material-0times-30min'].Density(table=((1e-09,
    ), ))
mdb.models['Model-1'].materials['Material-0times-30min'].Elastic(table=((
    0.22098, 0.3), ))
mdb.models['Model-1'].materials['Material-0times-30min'].Expansion(table=((
    0.018938, ), ))
mdb.models['Model-1'].materials['Material-0times-30min'].SpecificHeat(table=((
    896000000.0, ), ))
mdb.models['Model-1'].Material(name='Material-1times-30min')
mdb.models['Model-1'].materials['Material-1times-30min'].Conductivity(table=((
    167000.0, ), ))
mdb.models['Model-1'].materials['Material-1times-30min'].Density(table=((1e-09,
    ), ))
mdb.models['Model-1'].materials['Material-1times-30min'].Elastic(table=((
    1.3413, 0.3), ))
mdb.models['Model-1'].materials['Material-1times-30min'].Expansion(table=((
    0.021223, ), ))
mdb.models['Model-1'].materials['Material-1times-30min'].SpecificHeat(table=((
    896000000.0, ), ))
mdb.models['Model-1'].HomogeneousSolidSection(material='Material-0times-30min',
    name='Section-0times', thickness=None)
mdb.models['Model-1'].HomogeneousSolidSection(material='Material-1times-30min',
    name='Section-1times', thickness=None)
mdb.models['Model-1'].rootAssembly.DatumCsysByDefault(CARTESIAN)
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-1-1',
    part=mdb.models['Model-1'].parts['Part-1'])
del mdb.models['Model-1'].rootAssembly.features['Part-1-1']
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-1-1',
    part=mdb.models['Model-1'].parts['Part-1'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-2-1',
    part=mdb.models['Model-1'].parts['Part-2'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-3-1',
    part=mdb.models['Model-1'].parts['Part-3'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-4-1',
    part=mdb.models['Model-1'].parts['Part-4'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-5-1',
    part=mdb.models['Model-1'].parts['Part-5'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-6-1',
    part=mdb.models['Model-1'].parts['Part-6'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-7-1',
    part=mdb.models['Model-1'].parts['Part-7'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-8-1',
    part=mdb.models['Model-1'].parts['Part-8'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-9-1',
    part=mdb.models['Model-1'].parts['Part-9'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-10-1',
    part=mdb.models['Model-1'].parts['Part-10'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-11-1',
    part=mdb.models['Model-1'].parts['Part-11'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-12-1',
    part=mdb.models['Model-1'].parts['Part-12'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-13-1',
    part=mdb.models['Model-1'].parts['Part-13'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-14-1',
    part=mdb.models['Model-1'].parts['Part-14'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-15-1',
    part=mdb.models['Model-1'].parts['Part-15'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-16-1',
    part=mdb.models['Model-1'].parts['Part-16'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-17-1',
    part=mdb.models['Model-1'].parts['Part-17'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-18-1',
    part=mdb.models['Model-1'].parts['Part-18'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-19-1',
    part=mdb.models['Model-1'].parts['Part-19'])
mdb.models['Model-1'].rootAssembly.Instance(dependent=ON, name='Part-20-1',
    part=mdb.models['Model-1'].parts['Part-20'])
mdb.models['Model-1'].rootAssembly.instances['Part-2-1'].translate(vector=(
    45.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-3-1'].translate(vector=(
    51.0, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-4-1'].translate(vector=(
    56.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-5-1'].translate(vector=(
    62.0, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-6-1'].translate(vector=(
    67.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-7-1'].translate(vector=(
    73.0, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-8-1'].translate(vector=(
    78.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-9-1'].translate(vector=(
    84.0, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-10-1'].translate(vector=(
    89.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-11-1'].translate(vector=(
    95.0, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-12-1'].translate(vector=(
    100.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-13-1'].translate(vector=(
    106.0, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-14-1'].translate(vector=(
    111.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-15-1'].translate(vector=(
    117.0, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-16-1'].translate(vector=(
    122.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-17-1'].translate(vector=(
    128.0, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-18-1'].translate(vector=(
    133.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-19-1'].translate(vector=(
    139.0, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.instances['Part-20-1'].translate(vector=(
    144.5, 0.0, 0.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-2-1', ),
    vector=(-45.5, -2.5, 0.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-3-1', ),
    vector=(-51.0, 0.0, -10.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-4-1', ),
    vector=(-56.5, -2.5, -10.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-5-1', ),
    vector=(-62.0, 0.0, -20.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-6-1', ),
    vector=(-67.5, -2.5, -20.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-7-1', ),
    vector=(-73.0, 0.0, -30.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-8-1', ),
    vector=(-78.5, -2.5, -30.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-9-1', ),
    vector=(-84.0, 0.0, -40.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-10-1', ),
    vector=(-89.5, -2.5, -40.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-11-1', ),
    vector=(-95.0, 0.0, -50.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-12-1', ),
    vector=(-100.5, -2.5, -50.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-13-1', ),
    vector=(-106.0, 0.0, -60.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-14-1', ),
    vector=(-111.5, -2.5, -60.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-15-1', ),
    vector=(-117.0, 0.0, -70.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-16-1', ),
    vector=(-122.5, -2.5, -70.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-17-1', ),
    vector=(-128.0, 0.0, -80.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-18-1', ),
    vector=(-133.5, -2.5, -80.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-19-1', ),
    vector=(-139.0, 0.0, -90.0))
mdb.models['Model-1'].rootAssembly.translate(instanceList=('Part-20-1', ),
    vector=(-144.5, -2.5, -90.0))
mdb.models['Model-1'].rootAssembly.InstanceFromBooleanMerge(domain=GEOMETRY,
    instances=(mdb.models['Model-1'].rootAssembly.instances['Part-1-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-2-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-3-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-4-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-5-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-6-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-7-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-8-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-9-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-10-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-11-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-12-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-13-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-14-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-15-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-16-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-17-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-18-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-19-1'],
    mdb.models['Model-1'].rootAssembly.instances['Part-20-1']),
    keepIntersections=ON, name='Part-21', originalInstances=SUPPRESS)

data_numbers = 10000

for i in range(0, data_numbers):

    input_data = input_data_fu(data_length)
    np.savetxt('./FE_simulation_data/input_data/input_data' + str(i) + '.csv', input_data, delimiter=",", fmt="% s")

    jobname = 'Job-' + str(i)

    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask(('[#1 ]',
        ), ), name='Set-1')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-1'], sectionName=
        'Section-' + str(input_data[0][0]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask(('[#4 ]',
        ), ), name='Set-2')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-2'], sectionName=
        'Section-' + str(input_data[0][1]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask(('[#10 ]',
        ), ), name='Set-3')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-3'], sectionName=
        'Section-' + str(input_data[0][2]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask(('[#40 ]',
        ), ), name='Set-4')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-4'], sectionName=
        'Section-' + str(input_data[0][3]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#100 ]', ), ), name='Set-5')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-5'], sectionName=
        'Section-' + str(input_data[0][4]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#400 ]', ), ), name='Set-6')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-6'], sectionName=
        'Section-' + str(input_data[0][5]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#1000 ]', ), ), name='Set-7')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-7'], sectionName=
        'Section-' + str(input_data[0][6]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#4000 ]', ), ), name='Set-8')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-8'], sectionName=
        'Section-' + str(input_data[0][7]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#10000 ]', ), ), name='Set-9')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-9'], sectionName=
        'Section-' + str(input_data[0][8]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#40000 ]', ), ), name='Set-10')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-10'], sectionName=
        'Section-' + str(input_data[0][9]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask(('[#2 ]',
        ), ), name='Set-11')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-11'], sectionName=
        'Section-' + str(input_data[1][0]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask(('[#8 ]',
        ), ), name='Set-12')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-12'], sectionName=
        'Section-' + str(input_data[1][1]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask(('[#20 ]',
        ), ), name='Set-13')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-13'], sectionName=
        'Section-' + str(input_data[1][2]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask(('[#80 ]',
        ), ), name='Set-14')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-14'], sectionName=
        'Section-' + str(input_data[1][3]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#200 ]', ), ), name='Set-15')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-15'], sectionName=
        'Section-' + str(input_data[1][4]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#800 ]', ), ), name='Set-16')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-16'], sectionName=
        'Section-' + str(input_data[1][5]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#2000 ]', ), ), name='Set-17')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-17'], sectionName=
        'Section-' + str(input_data[1][6]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#8000 ]', ), ), name='Set-18')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-18'], sectionName=
        'Section-' + str(input_data[1][7]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#20000 ]', ), ), name='Set-19')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-19'], sectionName=
        'Section-' + str(input_data[1][8]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].parts['Part-21'].Set(cells=
        mdb.models['Model-1'].parts['Part-21'].cells.getSequenceFromMask((
        '[#80000 ]', ), ), name='Set-20')
    mdb.models['Model-1'].parts['Part-21'].SectionAssignment(offset=0.0,
        offsetField='', offsetType=MIDDLE_SURFACE, region=
        mdb.models['Model-1'].parts['Part-21'].sets['Set-20'], sectionName=
        'Section-' + str(input_data[1][9]) + 'times', thicknessAssignment=FROM_SECTION)
    mdb.models['Model-1'].rootAssembly.regenerate()
    mdb.models['Model-1'].CoupledTempDisplacementStep(deltmx=15.0, initialInc=0.01,
        minInc=1e-06, name='Step-1', nlgeom=ON, previous='Initial')
    mdb.models['Model-1'].rootAssembly.Set(cells=
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'].cells.getSequenceFromMask(
        ('[#fffff ]', ), ), edges=
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'].edges.getSequenceFromMask(
        ('[#ffffffff:4 #1ff ]', ), ), faces=
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'].faces.getSequenceFromMask(
        ('[#ffffffff:2 #fffffff ]', ), ), name='Set-1', vertices=
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'].vertices.getSequenceFromMask(
        ('[#ffffffff:2 #3 ]', ), ))
    mdb.models['Model-1'].TemperatureBC(createStepName='Initial', distributionType=
        UNIFORM, fieldName='', magnitude=0.0, name='BC-1', region=
        mdb.models['Model-1'].rootAssembly.sets['Set-1'])
    mdb.models['Model-1'].TabularAmplitude(data=((0.0, 0.0), (1.0, 1.0)), name=
        'Amp-1', smooth=SOLVER_DEFAULT, timeSpan=STEP)
    mdb.models['Model-1'].boundaryConditions['BC-1'].setValuesInStep(amplitude=
        'Amp-1', magnitude=100.0, stepName='Step-1')
    mdb.models['Model-1'].rootAssembly.Set(faces=
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'].faces.getSequenceFromMask(
        ('[#0:2 #4000000 ]', ), ), name='Set-2')
    mdb.models['Model-1'].ZsymmBC(createStepName='Initial', localCsys=None, name=
        'BC-2', region=mdb.models['Model-1'].rootAssembly.sets['Set-2'])
    mdb.models['Model-1'].rootAssembly.Set(faces=
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'].faces.getSequenceFromMask(
        ('[#0:2 #100000 ]', ), ), name='Set-3')
    mdb.models['Model-1'].ZsymmBC(createStepName='Initial', localCsys=None, name=
        'BC-3', region=mdb.models['Model-1'].rootAssembly.sets['Set-3'])
    mdb.models['Model-1'].rootAssembly.Set(edges=
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'].edges.getSequenceFromMask(
        ('[#0:4 #100 ]', ), ), name='Set-4')
    mdb.models['Model-1'].YsymmBC(createStepName='Initial', localCsys=None, name=
        'BC-4', region=mdb.models['Model-1'].rootAssembly.sets['Set-4'])
    mdb.models['Model-1'].rootAssembly.Set(edges=
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'].edges.getSequenceFromMask(
        ('[#0:4 #40 ]', ), ), name='Set-5')
    mdb.models['Model-1'].XsymmBC(createStepName='Initial', localCsys=None, name=
        'BC-5', region=mdb.models['Model-1'].rootAssembly.sets['Set-5'])
    mdb.models['Model-1'].rootAssembly.makeIndependent(instances=(
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'], ))
    mdb.models['Model-1'].rootAssembly.seedPartInstance(deviationFactor=0.1,
        minSizeFactor=0.1, regions=(
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'], ), size=1.0)
    mdb.models['Model-1'].rootAssembly.generateMesh(regions=(
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'], ))
    mdb.models['Model-1'].rootAssembly.setElementType(elemTypes=(ElemType(
        elemCode=C3D20RT, elemLibrary=STANDARD), ElemType(elemCode=UNKNOWN_WEDGE,
        elemLibrary=STANDARD), ElemType(elemCode=C3D10MT, elemLibrary=STANDARD)),
        regions=(
        mdb.models['Model-1'].rootAssembly.instances['Part-21-1'].cells.getSequenceFromMask(
        ('[#fffff ]', ), ), ))
    mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
        explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
        memory=90, memoryUnits=PERCENTAGE, model='Model-1', modelPrint=OFF,
        multiprocessingMode=DEFAULT, name=jobname, nodalOutputPrecision=SINGLE,
        numCpus=4, numDomains=4, numGPUs=1, numThreadsPerMpiProcess=1, queue=None,
        resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=
        0, waitMinutes=0)
    mdb.jobs[jobname].submit(consistencyChecking=OFF)
    mdb.jobs[jobname]._Message(STARTED, {'phase': BATCHPRE_PHASE,
        'clientHost': 'LAPTOP-12JUUN8V', 'handle': 0, 'jobName': jobname})
    mdb.jobs[jobname]._Message(WARNING, {'phase': BATCHPRE_PHASE,
        'message': 'THE ABSOLUTE ZERO TEMPERATURE HAS NOT BEEN SPECIFIED FOR COMPUTING INTERNAL THERMAL ENERGY USING THE ABSOLUTE ZERO PARAMETER ON THE *PHYSICAL CONSTANTS OPTION. A DEFAULT VALUE OF 0.0000 WILL BE ASSUMED.',
        'jobName': jobname})
    mdb.jobs[jobname]._Message(ODB_FILE, {'phase': BATCHPRE_PHASE,
        'file': '.\\FE_simulation_data\\simultaion_odb\\Job-'+ str(period) +'.odb', 'jobName': jobname})
    mdb.jobs[jobname]._Message(COMPLETED, {'phase': BATCHPRE_PHASE,
        'message': 'Analysis phase complete', 'jobName': jobname})
    mdb.jobs[jobname]._Message(STARTED, {'phase': STANDARD_PHASE,
        'clientHost': 'LAPTOP-12JUUN8V', 'handle': 19060, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STEP, {'phase': STANDARD_PHASE, 'stepId': 1,
        'jobName': jobname})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 0, 'jobName': jobname})
    mdb.jobs[jobname]._Message(MEMORY_ESTIMATE, {'phase': STANDARD_PHASE,
        'jobName': jobname, 'memory': 640.0})
    mdb.jobs[jobname]._Message(PHYSICAL_MEMORY, {'phase': STANDARD_PHASE,
        'physical_memory': 16252.0, 'jobName': jobname})
    mdb.jobs[jobname]._Message(MINIMUM_MEMORY, {'minimum_memory': 70.0,
        'phase': STANDARD_PHASE, 'jobName': jobname})
    mdb.jobs[jobname]._Message(WARNING, {'phase': STANDARD_PHASE,
        'message': 'The system matrix has 3 negative eigenvalues.',
        'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.01, 'attempts': 1,
        'timeIncrement': 0.01, 'increment': 1, 'stepTime': 0.01, 'step': 1,
        'jobName': jobname, 'severe': 0, 'iterations': 3, 'phase': STANDARD_PHASE,
        'equilibrium': 3})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 1, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.02, 'attempts': 1,
        'timeIncrement': 0.01, 'increment': 2, 'stepTime': 0.02, 'step': 1,
        'jobName': jobname, 'severe': 0, 'iterations': 2, 'phase': STANDARD_PHASE,
        'equilibrium': 2})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 2, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.03, 'attempts': 1,
        'timeIncrement': 0.01, 'increment': 3, 'stepTime': 0.03, 'step': 1,
        'jobName': jobname, 'severe': 0, 'iterations': 2, 'phase': STANDARD_PHASE,
        'equilibrium': 2})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 3, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.045, 'attempts': 1,
        'timeIncrement': 0.015, 'increment': 4, 'stepTime': 0.045, 'step': 1,
        'jobName': jobname, 'severe': 0, 'iterations': 2, 'phase': STANDARD_PHASE,
        'equilibrium': 2})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 4, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.0675, 'attempts': 1,
        'timeIncrement': 0.0225, 'increment': 5, 'stepTime': 0.0675, 'step': 1,
        'jobName': jobname, 'severe': 0, 'iterations': 2, 'phase': STANDARD_PHASE,
        'equilibrium': 2})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 5, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.10125, 'attempts': 1,
        'timeIncrement': 0.03375, 'increment': 6, 'stepTime': 0.10125, 'step': 1,
        'jobName': jobname, 'severe': 0, 'iterations': 3, 'phase': STANDARD_PHASE,
        'equilibrium': 3})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 6, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.151875, 'attempts': 1,
        'timeIncrement': 0.050625, 'increment': 7, 'stepTime': 0.151875, 'step': 1,
        'jobName': jobname, 'severe': 0, 'iterations': 4, 'phase': STANDARD_PHASE,
        'equilibrium': 4})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 7, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.2278125, 'attempts': 1,
        'timeIncrement': 0.0759375, 'increment': 8, 'stepTime': 0.2278125,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 5,
        'phase': STANDARD_PHASE, 'equilibrium': 5})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 8, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.30375, 'attempts': 1,
        'timeIncrement': 0.0759375, 'increment': 9, 'stepTime': 0.30375, 'step': 1,
        'jobName': jobname, 'severe': 0, 'iterations': 6, 'phase': STANDARD_PHASE,
        'equilibrium': 6})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 9, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.3796875, 'attempts': 1,
        'timeIncrement': 0.0759375, 'increment': 10, 'stepTime': 0.3796875,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 5,
        'phase': STANDARD_PHASE, 'equilibrium': 5})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 10, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.455625, 'attempts': 1,
        'timeIncrement': 0.0759375, 'increment': 11, 'stepTime': 0.455625,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 6,
        'phase': STANDARD_PHASE, 'equilibrium': 6})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 11, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.5315625, 'attempts': 1,
        'timeIncrement': 0.0759375, 'increment': 12, 'stepTime': 0.5315625,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 4,
        'phase': STANDARD_PHASE, 'equilibrium': 4})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 12, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.6075, 'attempts': 1,
        'timeIncrement': 0.0759375, 'increment': 13, 'stepTime': 0.6075, 'step': 1,
        'jobName': jobname, 'severe': 0, 'iterations': 4, 'phase': STANDARD_PHASE,
        'equilibrium': 4})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 13, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.72140625, 'attempts': 1,
        'timeIncrement': 0.11390625, 'increment': 14, 'stepTime': 0.72140625,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 8,
        'phase': STANDARD_PHASE, 'equilibrium': 8})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 14, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.72140625, 'attempts': ' 1U',
        'timeIncrement': 0.11390625, 'increment': 15, 'stepTime': 0.72140625,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 5,
        'phase': STANDARD_PHASE, 'equilibrium': 5})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.7498828125, 'attempts': 2,
        'timeIncrement': 0.0284765625, 'increment': 15, 'stepTime': 0.7498828125,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 2,
        'phase': STANDARD_PHASE, 'equilibrium': 2})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 15, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.778359375, 'attempts': 1,
        'timeIncrement': 0.0284765625, 'increment': 16, 'stepTime': 0.778359375,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 2,
        'phase': STANDARD_PHASE, 'equilibrium': 2})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 16, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.82107421875, 'attempts': 1,
        'timeIncrement': 0.04271484375, 'increment': 17, 'stepTime': 0.82107421875,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 2,
        'phase': STANDARD_PHASE, 'equilibrium': 2})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 17, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.885146484375, 'attempts': 1,
        'timeIncrement': 0.064072265625, 'increment': 18,
        'stepTime': 0.885146484375, 'step': 1, 'jobName': jobname, 'severe': 0,
        'iterations': 3, 'phase': STANDARD_PHASE, 'equilibrium': 3})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 18, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 0.9812548828125,
        'attempts': 1, 'timeIncrement': 0.0961083984375, 'increment': 19,
        'stepTime': 0.9812548828125, 'step': 1, 'jobName': jobname, 'severe': 0,
        'iterations': 5, 'phase': STANDARD_PHASE, 'equilibrium': 5})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 19, 'jobName': jobname})
    mdb.jobs[jobname]._Message(ODB_FRAME, {'phase': STANDARD_PHASE, 'step': 0,
        'frame': 20, 'jobName': jobname})
    mdb.jobs[jobname]._Message(STATUS, {'totalTime': 1.0, 'attempts': 1,
        'timeIncrement': 0.0187451171874999, 'increment': 20, 'stepTime': 1.0,
        'step': 1, 'jobName': jobname, 'severe': 0, 'iterations': 2,
        'phase': STANDARD_PHASE, 'equilibrium': 2})
    mdb.jobs[jobname]._Message(END_STEP, {'phase': STANDARD_PHASE, 'stepId': 1,
        'jobName': jobname})
    mdb.jobs[jobname]._Message(COMPLETED, {'phase': STANDARD_PHASE,
        'message': 'Analysis phase complete', 'jobName': jobname})
    mdb.jobs[jobname]._Message(JOB_COMPLETED, {'time': 'Sat Sep 23 18:13:24 2023',
        'jobName': jobname})
# Save by 24306 on 2023_09_23-18.14.20; build 2022 2021_09_16-01.57.30 176069
# Save by 24306 on 2023_09_23-18.17.26; build 2022 2021_09_16-01.57.30 176069
