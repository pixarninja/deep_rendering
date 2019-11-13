import maya.cmds as cmds
import maya.OpenMaya as om
import maya.OpenMayaUI as omui
from functools import partial

##########################
#### Helper Functions ####
##########################

# Convert world space to screen space
# https://video.stackexchange.com/questions/23382/maya-python-worldspace-to-screenspace-coordinates
def worldSpaceToScreenSpace(worldPoint):
    # Find the camera
    view = omui.M3dView.active3dView()
    cam = om.MDagPath()
    view.getCamera(cam)
    camPath = cam.fullPathName()
    
    # Get the dagPath to the camera shape node to get the world inverse matrix
    selList = om.MSelectionList()
    selList.add(cam)
    dagPath = om.MDagPath()
    selList.getDagPath(0,dagPath)
    dagPath.extendToShape()
    camInvMtx = dagPath.inclusiveMatrix().inverse()

    # Use a camera function set to get projection matrix, convert the MFloatMatrix 
    # into a MMatrix for multiplication compatibility
    fnCam = om.MFnCamera(dagPath)
    mFloatMtx = fnCam.projectionMatrix()
    projMtx = om.MMatrix(mFloatMtx.matrix)

    # Multiply all together and do the normalisation
    mPoint = om.MPoint(worldPoint[0],worldPoint[1],worldPoint[2]) * camInvMtx * projMtx;
    x = (mPoint[0] / mPoint[3] / 2 + .5)
    y = (mPoint[1] / mPoint[3] / 2 + .5)

    return [x,y]

# Collect all objects in the scene using Maya ls command
# https://stackoverflow.com/questions/22794533/maya-python-array-collecting
def collectObjects(currSel):
    meshSel = []
    for xform in currSel:
        shapes = cmds.listRelatives(xform, shapes=True) # it's possible to have more than one
        if shapes != None:
            for s in shapes:
                if cmds.nodeType(s) == 'mesh':
                    meshSel.append(xform)
  
    return meshSel
    
# Test if the mesh is bounded by the coordinates
# https://boomrigs.com/blog/2016/1/12/how-to-get-mesh-vertex-position-through-maya-api
def testMesh(mesh, bounds):    
    # Store bounds
    left = bounds[0][0]
    right = bounds[0][1]
    top = bounds[1][0]
    bottom = bounds[1][1]
    
    # Get Api MDagPath for object
    activeList = om.MSelectionList()
    activeList.add(mesh)
    dagPath = om.MDagPath()
    activeList.getDagPath(0, dagPath)

    # Iterate over all the mesh vertices and get position
    mItEdge = om.MItMeshEdge(dagPath)
    while not mItEdge.isDone():
        startPoint = mItEdge.point(0, om.MSpace.kWorld)
        endPoint = mItEdge.point(1, om.MSpace.kWorld)
        
        # Return with a True value if the edge is within the boundaries
        if clippingTest(startPoint, endPoint, bounds):
            return True
                
        mItEdge.next()
    
    return False
    
# Perform the Cohen-Sutherland Clipping test using Op Codes
# https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm
def clippingTest(p, q, bounds):
    opCodeP = opCode(worldSpaceToScreenSpace(p), bounds)
    opCodeQ = opCode(worldSpaceToScreenSpace(q), bounds)
        
    # Trivial reject
    if (opCodeP & opCodeQ):
        return False
        
    return True

# Return the Pp Code for a given point
def opCode(p, bounds):
    code = 0
    
    # Left of clipping window
    if p[0] < bounds[0][0]:
        code = code | 1
    
    # Right of clipping window
    if p[0] > bounds[0][1]:
        code = code | 2
        
    # Above clipping window
    if p[1] < bounds[1][0]:
        code = code | 4
        
    # Below clipping window
    if p[1] > bounds[1][1]:
        code = code | 8
        
    return code

##########################
### Main Functionality ###
##########################

# Create and display menu system
def displayWindow():
    menu = cmds.window( title="Export Semantics Tool", iconName='SemanticsTool', widthHeight=(350, 400) )
    scrollLayout = cmds.scrollLayout( verticalScrollBarThickness=16 )
    cmds.flowLayout( columnSpacing=10 )
    cmds.columnLayout( cat=('both', 25), rs=10, cw=340 )
    cmds.text( label="\nThis is the \"Sematics Tool\"! This tool will generate semantics for the loaded scene.\n\n", ww=True, al="left" )
    cmds.text( label="To run:\n1) Input the information in the fields below.\n2) Click \"Run\".", al="left" )
    cmds.text( '1', label='Enter the keyframe at which to start semantics generation:', al='left', ww=True )
    startTimeField = cmds.textField()
    cmds.text( '120', label='Enter the keyframe at which to end semantics generation:', al='left', ww=True )
    endTimeField = cmds.textField()
    cmds.text( '1', label='Enter the step at which to process frames:', al='left', ww=True )
    stepTimeField = cmds.textField()
    cmds.text( '1', label='Enter the dimension of frame blocks, or -1 for the whole frame:', al='left', ww=True )
    blockDimField = cmds.textField()
    cmds.button( label='Run', command=partial( generateSemantics, menu, startTimeField, endTimeField, stepTimeField, blockDimField ) )
    cmds.text( label="\n", al='left' )
    cmds.showWindow( menu )

def generateSemantics( menu, startTimeField, endTimeField, stepTimeField, blockDimField, *args ):
    # Grab user input and delete window
    startTime = cmds.textField(startTimeField, q=True, tx=True )
    if (startTime == ''):
        print 'WARNING: Default start time (1) used...'
        startTime = '1'
    endTime = cmds.textField(endTimeField, q=True, tx=True )
    if (endTime == ''):
        print 'WARNING: Default end time (120) used...'
        endTime = '120'
    stepTime = cmds.textField(stepTimeField, q=True, tx=True )
    if (stepTime == ''):
        print 'WARNING: Default step time (1) used...'
        stepTime = '1'
    blockDim = cmds.textField(blockDimField, q=True, tx=True )
    if (blockDim == ''):
        print 'WARNING: Default block dim (-1) used...'
        blockDim = '-1'
    cmds.deleteUI( menu, window=True )
    
    # Set up program
    resWidth = cmds.getAttr('defaultResolution.width')
    resHeight = cmds.getAttr('defaultResolution.height')
    print(resWidth, resHeight)
    blocks = []
    if int(blockDim) < 0:
        blocks.append([[0,1],[0,1]])
    else:
        for w in range(0, resWidth / int(blockDim)):
            left = (w * int(blockDim)) / float(resWidth)
            right = ((w + 1) * int(blockDim)) / float(resWidth)
            for h in range(0, resHeight / int(blockDim)):
                top = (h * int(blockDim)) / float(resHeight)
                bottom = ((h + 1) * int(blockDim)) / float(resHeight)
                
                blocks.append([[left,right],[top,bottom]])
    
    # Obtain all meshes in the scene
    currSel = cmds.ls()
    meshes = collectObjects(currSel)
    
    # Iterate over all meshes and all boundaries
    for bounds in blocks:
        print(bounds)
        for i, mesh in enumerate(meshes):
            inBounds = testMesh(mesh, bounds)
            if inBounds:
                print(mesh)
    
##########################
####### Run Script #######
##########################

# Display window
displayWindow()