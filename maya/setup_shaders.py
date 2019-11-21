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
    mPoint = om.MPoint(worldPoint[0],worldPoint[1],worldPoint[2]) * camInvMtx * projMtx
    x = (mPoint[0] / mPoint[3] / 2 + .5)
    y = 1 - (mPoint[1] / mPoint[3] / 2 + .5)
    
    return [x,y]

# Collect all objects in the scene using Maya ls command
# https://stackoverflow.com/questions/22794533/maya-python-array-collecting
def collectObjects(currSel):
    meshSel = []
    for xform in currSel:
        shapes = cmds.listRelatives(xform, shapes=True) # it's possible to have more than one
        if shapes is not None:
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
    P = worldSpaceToScreenSpace(p)
    Q = worldSpaceToScreenSpace(q)
    opCodeP = opCode(P, bounds)
    opCodeQ = opCode(Q, bounds)
        
    # Trivial reject
    if (opCodeP & opCodeQ):
        return False
        
    return True

# Return the Op Code for a given point
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
    
# Update the color of a shader given r, g, b
def updateShaderColor(mesh, colorCode, n):
    shader = findShader(mesh)
    cmds.setAttr ( (shader) + '.r', colorCode[0] )
    cmds.setAttr ( (shader) + '.g', colorCode[1] )
    cmds.setAttr ( (shader) + '.b', colorCode[2] ) 
    cmds.setAttr ( (shader) + '.n', n ) 
    
# Return correct shader given a shader name
def findShader(mesh):
    cmds.select(mesh)
    nodes = cmds.ls(sl=True, dag=True, s=True)
    shadingEngine = cmds.listConnections(nodes, type='shadingEngine')
    materials = cmds.ls(cmds.listConnections(shadingEngine), materials=True)
    
    # Find the OSL shader node from connected nodes of the material
    for node in cmds.listConnections(materials):
        if node.find('PxrOSL') > -1:
            return node
    return None

##########################
### Main Functionality ###
##########################

# Create and display menu system
def displayWindow():
    menu = cmds.window( title="Setup Semantics Tool", iconName='SetupSemanticsTool', widthHeight=(350, 400) )
    scrollLayout = cmds.scrollLayout( verticalScrollBarThickness=16 )
    cmds.flowLayout( columnSpacing=10 )
    cmds.columnLayout( cat=('both', 25), rs=10, cw=340 )
    cmds.text( label="\nThis is the \"Semantics Shader Tool\"! This tool will generate semantics shaders for the loaded scene.\n\n", ww=True, al="left" )
    cmds.text( label="To run:\n1) Input the information in the fields below.\n2) Click \"Run\".", al="left" )
    cmds.text( label='Enter the keyframe at which to start semantics generation (1):', al='left', ww=True )
    startTimeField = cmds.textField()
    cmds.text( label='Enter the keyframe at which to end semantics generation (120):', al='left', ww=True )
    endTimeField = cmds.textField()
    cmds.text( label='Enter the step at which to process frames (1):', al='left', ww=True )
    stepTimeField = cmds.textField()
    cmds.text( label='Enter the number of bits used to store each, a multiple of 8 is recommended (8):', al='left', ww=True )
    bitNumField = cmds.textField()
    cmds.button( label='Run', command=partial( setupShaders, menu, startTimeField, endTimeField, stepTimeField, bitNumField ) )
    cmds.text( label="\n", al='left' )
    cmds.showWindow( menu )

def setupShaders( menu, startTimeField, endTimeField, stepTimeField, bitNumField, *args ):
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
    bitNum = cmds.textField(bitNumField, q=True, tx=True )
    if (bitNum == ''):
        print 'WARNING: Default bit number (8) used...'
        bitNum = '8'
    N = int(bitNum)
    cmds.deleteUI( menu, window=True )
    
    # Set up program
    resWidth = cmds.getAttr('defaultResolution.width')
    resHeight = cmds.getAttr('defaultResolution.height')
    blockDim = [int(resWidth / (2 * N)), int(resHeight / ((N / 8) * N))]
    xDiv = float(resWidth) / blockDim[0]
    yDiv = float(resHeight) / blockDim[1]
    step = (resWidth / blockDim[0]) / (N / 2)
        
    # Set up blocks
    blocks = []
    for h in range(int(yDiv)):
        row = []
        
        # Find boundaries for each block in the row
        top = h / yDiv
        bottom = (h + 1) / yDiv
        for w in range(int(xDiv)):
            left = w / xDiv
            right = (w + 1) / xDiv
            
            row.append([[left,right],[top,bottom]])
            
        # Append the finished row
        blocks.append(row)
            
    print('Block Dim: (%d, %d), Blocks: (%d, %d)' % (blockDim[0], blockDim[1], len(blocks), len(blocks[0])))
    
    # Obtain all meshes in the scene
    currSel = cmds.ls()
    meshes = collectObjects(currSel)
    meshColors = []
    for n in range(len(meshes)):
        meshColors.append([0x0, 0x0, 0x0])
    
    # Iterate over all meshes and all boundaries
    for k, mesh in enumerate(meshes):
        cmds.select(mesh)
        bb = cmds.xform( q=True, bb=True, ws=True )
        
        # Obtain all 8 points to test from the bounding box
        # Format: xmin ymin zmin xmax ymax zmax
        bbPoints = []
        bbPoints.append(om.MPoint( bb[0], bb[1], bb[2], 1.0 ))
        bbPoints.append(om.MPoint( bb[0], bb[1], bb[5], 1.0 ))
        bbPoints.append(om.MPoint( bb[0], bb[4], bb[2], 1.0 ))
        bbPoints.append(om.MPoint( bb[0], bb[4], bb[5], 1.0 ))
        bbPoints.append(om.MPoint( bb[3], bb[1], bb[2], 1.0 ))
        bbPoints.append(om.MPoint( bb[3], bb[1], bb[5], 1.0 ))
        bbPoints.append(om.MPoint( bb[3], bb[4], bb[2], 1.0 ))
        bbPoints.append(om.MPoint( bb[3], bb[4], bb[5], 1.0 ))
        
        # Translate to screen space and obtain overall bounds
        left, right, top, bottom = 1.0, 0.0, 1.0, 0.0
        for p in bbPoints:
            P = worldSpaceToScreenSpace(p)
            if left > P[0]:
                left = P[0]
            if right < P[0]:
                right = P[0]
            if top > P[1]:
                top = P[1]
            if bottom < P[1]:
                bottom = P[1]
                    
        if left < 0.0 or left >= 1.0:
            left = 0.0
        if right > 1.0 or right <= 0.0:
            right = 1.0
        if top < 0.0 or top >= 1.0:
            top = 0.0
        if bottom > 1.0 or bottom <= 0.0:
            bottom = 1.0
        
        # Translate bounds to i and j values
        bounds = [int(left * len(blocks[0])), int(right * len(blocks[0])) + 1, int(top * len(blocks)), int(bottom * len(blocks)) + 1]
        if bounds[0] > len(blocks[0]) - 1:
            bounds[0] = len(blocks[0]) - 1
        if bounds[1] > len(blocks[0]) - 1:
            bounds[1] = len(blocks[0]) - 1
        if bounds[2] > len(blocks) - 1:
            bounds[2] = len(blocks) - 1
        if bounds[3] > len(blocks) - 1:
            bounds[3] = len(blocks) - 1
        
        print('Processing {}: [({},{}),({},{})]'.format(mesh, bounds[0], bounds[1], bounds[2], bounds[3]))
        
        for i in range(bounds[2], bounds[3] + 1):
            b = i % N
            for j in range(bounds[0], bounds[1] + 1):
                r = j % N
                g = int((i / (N / 2))) * int(step) + int((j / (N / 2)))
                
                # Find bounds and color code for current block
                subBounds = blocks[i][j]
                colorCode = [0x1 << r, 0x1 << g, 0x1 << b]
                
                # Test which meshes are contained within the block
                if testMesh(mesh, subBounds):
                    for n in range(len(colorCode)):
                        meshColors[k][n] |= colorCode[n]

    for k, mesh in enumerate(meshes):
        updateShaderColor(mesh, meshColors[k], N)
        print(mesh, meshColors[k])
    
##########################
####### Run Script #######
##########################

# Display window
displayWindow()