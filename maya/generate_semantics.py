import maya.cmds as cmds
import maya.OpenMaya as om
import maya.OpenMayaUI as omui
from functools import partial
import json as json
import os as os
import re as re

class MeshInfo:
    def __init__(self, mesh, bbox):
        self.mesh = mesh
        self.bbox = bbox

##########################
#### Helper Functions ####
##########################

# Convert screen space to world space
# https://forums.autodesk.com/t5/maya-programming/getting-click-position-in-world-coordinates/td-p/7578289
def screenSpaceToWorldSpace(screenPoint):
    worldPos = om.MPoint() # out variable
    worldDir = om.MVector() # out variable
    
    activeView = omui.M3dView().active3dView()
    activeView.viewToWorld(int(screenPoint[0]), int(screenPoint[1]), worldPos, worldDir)
    
    return worldPos

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
    cmds.select( cl=True )
    for xform in currSel:
        shapes = cmds.listRelatives(xform, shapes=True) # it's possible to have more than one
        if shapes is not None:
            for s in shapes:
                if cmds.nodeType(s) == 'mesh':
                    meshSel.append( xform )
                    cmds.select( xform, add=True )
  
    return meshSel

# Return the bounding box for a given mesh
def findBoundingBox(mesh):
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
    
    return [[left, right], [top, bottom]]

# Test if the mesh is bounded by the coordinates
# https://boomrigs.com/blog/2016/1/12/how-to-get-mesh-vertex-position-through-maya-api
def findFaces(meshInfo, bounds):
    # Extract mesh information
    mesh = meshInfo.mesh
    bbox = meshInfo.bbox
    faces = []
    
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
        
        # Track the edge and its faces if it is within the bounds
        p, q = clippingTest(startPoint, endPoint, bounds)
        if not (p & q):
            faceEdges = om.MIntArray()
            mItEdge.getConnectedFaces( faceEdges )
            if faceEdges not in faces:
                faces.append(faceEdges)
                
        mItEdge.next()

    meshInfo.faces = faces
    return meshInfo
    
def calculatePrecisionValue(meshInfo, bounds, subDim):
    # Calculate maximum value
    maximum = 0x1
    for i in range(1, subDim * (subDim - 1) + subDim):
        maximum |= 2**i

    # Short-circuit for special cases
    if 'Floor' in meshInfo.mesh:
        return maximum

    # Store bounds
    left = bounds[0][0]
    right = bounds[0][1]
    top = bounds[1][0]
    bottom = bounds[1][1]
    blockDim = (right - left) / float(subDim)

    # Calculate precision value, v
    v = 0
    if len(meshInfo.faces) > 0:
        for m in range(0, subDim):
            for n in range(0, subDim):
                k = 0x1 << (subDim * m + n)
                subBounds = [[left + n * blockDim, left + (n + 1) * blockDim] , [top + m * blockDim, top + (m + 1) * blockDim]]

                if testFaces(meshInfo, subBounds):
                    # Short circuit if the max value is reached
                    if v | k == maximum:
                        return maximum
                    
                    v |= k

    return v

# Iterate over edges found in each face
def testFaces(meshInfo, bounds):
    mesh = meshInfo.mesh
    faces = meshInfo.faces
    
    for facePair in faces:
        for face in facePair:
            cmds.select( '{}.f[{}]'.format(mesh, face), r=True )
            edgeData = cmds.polyListComponentConversion(cmds.ls( sl=True ), ff=True, te=True)
            edges = []
            for data in edgeData:
                if ':' in data:
                    seq = data.split(':')
                    start = int(re.sub("[^0-9]", "", seq[0]))
                    end = int(re.sub("[^0-9]", "", seq[1]))
                    for e in range(start, end + 1):
                        edges.append(e)
                else:
                    edges.append(re.sub("[^0-9]", "", data))

            codes = 0
            for edge in edges:
                cmds.select( cl=True )
                cmds.select(mesh +'.e[{}]'.format(edge), add=True)
                cmds.select( cmds.polyListComponentConversion( tv=True ) ) 
                verts = cmds.ls( sl=True )
                vertPositions = cmds.xform(verts, q=True, ws=True, t=True)
                startPoint, endPoint = vertPositions[0:3], vertPositions[3:7]
                            
                # Check if the edge is within the boundaries
                p, q = clippingTest(startPoint, endPoint, bounds)
                
                # Update codes variable to check for face coverage
                if not (p & q):
                    codes = 0xf
                else:
                    codes |= (p | q)
                    
                # Test face-cover edge case
                if codes == 0xf:
                    return True

    return False
    
# Perform the Cohen-Sutherland Clipping test using Op Codes
# https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm
def clippingTest(p, q, bounds):
    P = worldSpaceToScreenSpace(p)
    Q = worldSpaceToScreenSpace(q)
    opCodeP = opCode(P, bounds)
    opCodeQ = opCode(Q, bounds)

    return opCodeP, opCodeQ
        
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

# Extract semantic data based on block position and meshes in block
def extractSemantics(meshesInBlock, screenPoint):
    semantics = []
    for mesh in meshesInBlock[0]:
        semanticsForMesh = ''
        v = meshesInBlock[0][mesh]
        worldPoint = screenSpaceToWorldSpace(screenPoint)
        mesh = mesh.replace('Shape', '')
        d = postionDistance(meshPosition(mesh), worldPoint)
        
        # Text formatting
        divider = ', '
        semanticsForMesh += '{}_d{:d}'.format( mesh, int(d * 1e5) )
        semanticsForMesh += divider + '{}_v{:d}'.format( mesh, int(v) )

        if semanticsForMesh not in semantics:
            semantics.append(semanticsForMesh)
    
    return semantics
    
def formatList(list):
    for i, val in enumerate(list):
        if float('{0:.6f}'.format( val )) == -0.0:
            list[i] = 0.0
            
    return list

# Return the Euclidean distance between the centers of two meshes
def findDistance(meshA, meshB):
    return postionDistance(meshPosition(meshA), meshPosition(meshB))

# Obtain the position of a mesh in world space
def meshPosition(mesh):
    cmds.select(mesh)
    return cmds.xform( q=True, ws=True, t=True )

# Find the distance between two points
def postionDistance(posA, posB):
    return ((posA[0] - posB[0])**2 + (posA[1] - posB[1])**2 + (posA[2] - posB[2])**2)**0.5

# Makes a folder if none exists for the input path
# https://stackoverflow.com/questions/47738227/create-a-folder-using-python-in-a-maya-program
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

##########################
### Main Functionality ###
##########################

# Create and display menu system
def displayWindow():
    menu = cmds.window( title="Generate Semantics Tool", iconName='GenerateSemanticsTool', widthHeight=(350, 400) )
    scrollLayout = cmds.scrollLayout( verticalScrollBarThickness=16 )
    cmds.flowLayout( columnSpacing=10 )
    cmds.columnLayout( cat=('both', 25), rs=10, cw=340 )
    cmds.text( label="\nThis is the \"Generate Sematics Tool\"! This tool will generate semantics for the loaded scene.\n\n", ww=True, al="left" )
    cmds.text( label="To run:\n1) Input the information in the fields below.\n2) Click \"Run\".", al="left" )
    cmds.text( label='Enter the keyframe at which to start semantics generation (1):', al='left', ww=True )
    startTimeField = cmds.textField()
    cmds.text( label='Enter the keyframe at which to end semantics generation (1):', al='left', ww=True )
    endTimeField = cmds.textField()
    cmds.text( label='Enter the step at which to process frames (1):', al='left', ww=True )
    stepTimeField = cmds.textField()
    cmds.text( label='Enter the precision for generating semantics (4):', al='left', ww=True )
    precisionField = cmds.textField()
    cmds.text( label='Enter the dimension for each extracted image, a power of 2 is recommended (64):', al='left', ww=True )
    dimensionField = cmds.textField()
    cmds.button( label='Run', command=partial( generateSemantics, menu, startTimeField, endTimeField, stepTimeField, precisionField, dimensionField ) )
    cmds.text( label="\n", al='left' )
    cmds.showWindow( menu )

def generateSemantics( menu, startTimeField, endTimeField, stepTimeField, precisionField, dimensionField, *args ):
    # Grab user input and delete window
    startTime = cmds.textField(startTimeField, q=True, tx=True )
    if (startTime == ''):
        print 'WARNING: Default start time (1) used...'
        startTime = '1'
    endTime = cmds.textField(endTimeField, q=True, tx=True )
    if (endTime == ''):
        print 'WARNING: Default end time (1) used...'
        endTime = '1'
    stepTime = cmds.textField(stepTimeField, q=True, tx=True )
    if (stepTime == ''):
        print 'WARNING: Default step time (1) used...'
        stepTime = '1'
    precision = cmds.textField(precisionField, q=True, tx=True )
    if (precision == '' or int(precision) < 0):
        print 'WARNING: Default precision (4) used...'
        precision = '4'
    dimension = cmds.textField(dimensionField, q=True, tx=True )
    if (dimension == ''):
        print 'WARNING: Default dimension (64) used...'
        dimension = '64'
    dim = int(dimension)
    subdiv = int(precision)
    maximum = 0x1
    for i in range(1, subdiv * (subdiv - 1) + subdiv):
        maximum |= 2**i
    cmds.deleteUI( menu, window=True )
    cmds.currentTime( int(startTime), edit=True )
    currTime = startTime
    
    # Set up program
    print('Disabling UNDO to save memory usage...')
    #cmds.undoInfo( state=False )
    resWidth = cmds.getAttr('defaultResolution.width')
    resHeight = cmds.getAttr('defaultResolution.height')
    blockDim = [dim, dim]
    xDiv = float(resWidth) / blockDim[0]
    yDiv = float(resHeight) / blockDim[1]

    print(resWidth, resHeight, blockDim, xDiv, yDiv)
    
    # Iterate over range of frames
    while int(currTime) <= int(endTime):
        print('Processing frame {:03d}'.format(int(currTime)))

        # Obtain all meshes in the scene
        currSel = cmds.ls()
        meshes = collectObjects(currSel)
        cmds.selectMode( co=True )
    
        # Set up blocks and view
        blockToMeshMap = []
        view = omui.M3dView.active3dView()
        om.MGlobal.setSelectionMode(om.MGlobal.kSelectComponentMode)
        comMask = om.MSelectionMask(om.MSelectionMask.kSelectMeshFaces)
        rw = view.portWidth() / float(resWidth)
        rh = view.portHeight() / float(resHeight)

        for h in range(int(yDiv)):
            row = []
            blockToMeshMap.append([])
            
            # Find boundaries for each block in the row
            top = h * rh * blockDim[1]
            bottom = (h + 1) * rh * blockDim[1]
            for w in range(int(xDiv)):
                blockToMeshMap[h].append([])
                left = w * rw * blockDim[0]
                right = (w + 1) * rw * blockDim[0]
                subBlockDim = (right - left) / float(subdiv)
                row.append([[left,right],[top,bottom]])

                for m in range(0, subdiv):
                    for n in range(0, subdiv):
                        v = 0x1 << (subdiv * m + n)
                        subBounds = [[left + n * subBlockDim, left + (n + 1) * subBlockDim] , [top + m * subBlockDim, top + (m + 1) * subBlockDim]]

                        # https://forums.cgsociety.org/t/api-selectfromscreen/1590726/2                     
                        om.MGlobal.setComponentSelectionMask(comMask)
                        om.MGlobal.selectFromScreen( int(subBounds[0][0]), view.portHeight() - int(subBounds[1][0]), int(subBounds[0][1]), view.portHeight() - int(subBounds[1][1]), om.MGlobal.kReplaceList, om.MGlobal.kWireframeSelectMethod)
                        
                        objects = om.MSelectionList()
                        om.MGlobal.getActiveSelectionList(objects)
                        fromScreen = []
                        objects.getSelectionStrings(fromScreen)
                        
                        for face in fromScreen:
                            mesh = face.split('.')[0]
                            if not blockToMeshMap[h][w]:
                                blockToMeshMap[h][w].append({ mesh: v })
                            elif mesh not in blockToMeshMap[h][w][0]:
                                blockToMeshMap[h][w][0][mesh] = v
                            else:
                                blockToMeshMap[h][w][0][mesh] = blockToMeshMap[h][w][0][mesh] | v
                            
        # Extract semantics for each mesh
        framePath = 'C:\\Users\\wesha\\Git\\deep_rendering\\python\\datasets\\Frame\\training\\{}\\attributes\\{:03d}'.format( dim, int(currTime) )
        make_dir(framePath)
        for i, data in enumerate(blockToMeshMap):
            for j, meshesInBlock in enumerate(data):
                if meshesInBlock:
                    # Screen point = Ydim * (i + 1), Xdim * (j + 1)
                    screenPoint = blockDim[1] * (i + 0.5), blockDim[0] * (j + 0.5)
                    semantics = extractSemantics(meshesInBlock, screenPoint)

                    #print('Processing semantics for block({},{})'.format( i, j ))
                    blockPath = framePath + '\\{:03d}.txt'.format( int(i * xDiv + j + 1) )
                    with open(blockPath, 'w') as f:
                        f.write( ' and '.join( semantics ).replace('|', '') + '\n' )
        
        # Move to next frame
        cmds.currentTime( int(currTime) + int(stepTime), edit=True )
        currTime = int(currTime) + int(stepTime)
    
##########################
####### Run Script #######
##########################

# Display window
displayWindow()