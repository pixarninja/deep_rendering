import maya.cmds as cmds
import maya.OpenMaya as om
import maya.OpenMayaUI as omui
from functools import partial
import json as json
import os as os
import re as re
import math as math

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
def extractSemantics(meshes, meshesInBlock, screenPoint, frameOffset):
    semantics = []
    firstMesh = True

    # Store frames, offset could be calculated in a loop
    f = getValue('f', meshesInBlock[0])

    for mesh in meshes:
        mesh = mesh.replace('Shape', '')
        semanticsForMesh = None
        
        #if 'Floor' not in mesh:# or (f is not None and int(f) % 11 == 0):
        # Store precision
        v = getValue('v_{}'.format(mesh), meshesInBlock[0])

        # Store normals
        n = getValue('n_{}'.format(mesh), meshesInBlock[0])

        # Store rotation
        r = getValue('r_{}'.format(mesh), meshesInBlock[0])

        # Store block
        b = getValue('b_{}'.format(mesh), meshesInBlock[0])

        # Store selected faces
        i = getValue('i_{}'.format(mesh), meshesInBlock[0])

        # Store distance
        worldPoint = screenSpaceToWorldSpace(screenPoint)
        d = postionDistance(meshPosition(mesh), worldPoint)

        # Store frame for the first mesh processed
        if f is not None and firstMesh:
            firstMesh = False
            fc = int(f)
            fp = fc - frameOffset
            fn = fc + frameOffset
            # if fp >= 0:
            #     semantics.append('fp:{} and f:{} and fn:{}'.format( fp, fc, fn ))
            # else:
            #     semantics.append('f:{} and fn:{}'.format( fc, fn ))
            if fp != fn:
                semantics.append('f:{} and f:{}'.format( fp, fn ))
            else:
                semantics.append('f:{}'.format( fc ))

        # Skip if the mesh has no semantics to add
        if v is None:
            continue
        
        # Text formatting, with special case for Floor
        prefix = '{}.d:{:06d}'.format( mesh, int(d * 1e5) )
        divider = ' and '
        if b is not None:
            prefix = '{} with b:{}'.format( mesh, b )

        # Store initial semantics
        semanticsForMesh = prefix

        # Process face indices
        if i is not None:
            semanticsForMesh += divider + '{}.i:{}'.format( mesh, i )

        if 'Floor' not in mesh:

            # Process precision
            if v is not None:
                semanticsForMesh += divider + '{}.v:{}'.format( mesh, int(v) )

            # Process normals
            if n is not None:
                # attr = ''
                # div = '-'
                # epsilon = 30
                # if n.x > epsilon:
                #     attr += 'r' # right
                # elif n.x < -epsilon:
                #     attr += 'l' # left
                # else:
                #     attr += 'c' # center
                # attr += div
                # if n.y > epsilon:
                #     attr += 'o' # over
                # elif n.y < -epsilon:
                #     attr += 'u' # under
                # else:
                #     attr += 'c' # center
                # attr += div
                # if n.z > epsilon:
                #     attr += 'f' # front
                # elif n.z < -epsilon:
                #     attr += 'b' # back
                # else:
                #     attr += 'c' # center
                # semanticsForMesh += divider + '{}.n:{}'.format( mesh, attr )
                nx = cmds.angleBetween( euler=True, v1=(1.0, 0, 0), v2=(n.x, n.y, n.z) )
                ny = cmds.angleBetween( euler=True, v1=(0, 1.0, 0), v2=(n.x, n.y, n.z) )
                nz = cmds.angleBetween( euler=True, v1=(0, 0, 1.0), v2=(n.x, n.y, n.z) )
                semanticsForMesh += divider + '{}.n:{}-{}-{}'.format( mesh, int(nx[0]), int(ny[1]), int(nz[2]) )

            # Process rotation
            if r is not None:
                semanticsForMesh += divider + '{}.r:{}-{}-{}'.format( mesh, int(r[0]), int(r[1]), int(r[2]) )

        if semanticsForMesh not in semantics:
            semantics.append(semanticsForMesh)
    
    return semantics

def getValue(key, dictionary):
    if key in dictionary:
        return dictionary[key]
    return None
    
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
    cmds.text( label='Enter the frame to generate test captions for (1):', al='left', ww=True )
    testTimeField = cmds.textField()
    cmds.text( label='Enter the precision for generating semantics (4):', al='left', ww=True )
    precisionField = cmds.textField()
    cmds.text( label='Enter the dimension for each extracted image, a power of 2 is recommended (64):', al='left', ww=True )
    dimensionField = cmds.textField()
    cmds.text( label='Enter the offset for each block, a value >1 blocks will overlap (1):', al='left', ww=True )
    offsetField = cmds.textField()
    cmds.button( label='Run', command=partial( generateSemantics, menu, startTimeField, endTimeField, stepTimeField, testTimeField, precisionField, dimensionField, offsetField ) )
    cmds.text( label="\n", al='left' )
    cmds.showWindow( menu )

def generateSemantics( menu, startTimeField, endTimeField, stepTimeField, testTimeField, precisionField, dimensionField, offsetField, *args ):
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
    testTime = cmds.textField(testTimeField, q=True, tx=True )
    if (testTime == ''):
        print 'WARNING: Default test time (1) used...'
        testTime = '1'
    precision = cmds.textField(precisionField, q=True, tx=True )
    if (precision == '' or int(precision) < 0):
        print 'WARNING: Default precision (4) used...'
        precision = '4'
    dimension = cmds.textField(dimensionField, q=True, tx=True )
    if (dimension == ''):
        print 'WARNING: Default dimension (64) used...'
        dimension = '64'
    offset = cmds.textField(offsetField, q=True, tx=True )
    if (offset == ''):
        print 'WARNING: Default offset (1) used...'
        offset = '1'
    dim = int(dimension)
    subdiv = int(precision)
    blockOffset = float(offset)
    postfix = ''
    if blockOffset != 1:
        if int(blockOffset) == float(blockOffset):
            postfix = '_{}'.format(int(blockOffset))
        else:
            postfix = '_{}'.format(float(blockOffset)).replace('.', '-')
    maximum = 0x1
    for i in range(1, subdiv * (subdiv - 1) + subdiv):
        maximum |= 2**i
    cmds.deleteUI( menu, window=True )
    cmds.currentTime( int(startTime), edit=True )
    currTime = startTime
    
    # Set up program
    print('Disabling UNDO to save memory usage...')
    cmds.undoInfo( state=False )
    resWidth = cmds.getAttr('defaultResolution.width')
    resHeight = cmds.getAttr('defaultResolution.height')
    blockDim = [dim, dim]
    xDiv = float(resWidth) / blockDim[0] * blockOffset
    yDiv = float(resHeight) / blockDim[1] * blockOffset
    faceListToIdMap = []
    testIndices = []

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
        pw = view.portWidth()
        ph = view.portHeight()
        rw = pw / float(resWidth)
        rh = ph / float(resHeight)
        epsilon = 1e-4
        print(pw, ph, rw, rh, epsilon)

        # Create sliding window.
        left = 0
        right = blockDim[0] * rw
        top = 0
        bottom = blockDim[1] * rh

        # Find the Region Of Interest (ROI).
        #for h in range(int(yDiv)):
        while bottom < ph or abs(bottom - ph) < epsilon:
            row = []
            blockToMeshMap.append([])
            h = len(blockToMeshMap) - 1

            #for w in range(int(xDiv)):
            while right < pw or abs(right - pw) < epsilon:
                blockToMeshMap[h].append([])
                w = len(blockToMeshMap[h]) - 1

                subBlockDim = (right - left) / float(subdiv)
                row.append([[left,right],[top,bottom]])

                # If at test frame, record the index if it's a standard block.
                if (currTime == testTime) and (int(round(left / rw, 0)) % blockDim[0] == 0) and (int(round(right / rw, 0)) % blockDim[0] == 0) and (int(round(top / rh, 0)) % blockDim[1] == 0) and (int(round(bottom / rh, 0)) % blockDim[1] == 0):
                    testIndices.append((h, w))

                faceIndices = {}
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
                        
                        # Store tuples of data and insert into dictionary
                        data = []
                        for face in fromScreen:
                            mesh = face.split('.')[0].replace('Shape', '')

                            # Store all the indices in the selected faces.
                            for faceIndex in face.split(':'):
                                index = int( re.sub('[^0-9]', '', faceIndex) )

                                if not faceIndices or mesh not in faceIndices:
                                    faceIndices[mesh] = [index]
                                elif index not in faceIndices[mesh]:
                                    faceIndices[mesh].append(index)

                            # Store precision information
                            data.append((mesh, 'v', v))

                            # Store normal information
                            # query = [float(q) for q in cmds.xform( face, q=True, matrix=True, ws=True )]
                            # matrix = om.MMatrix()
                            # om.MScriptUtil().createMatrixFromList(query, matrix)
                            # cmds.select( face )
                            # normalInfo = cmds.polyInfo( fn=True )
                            # n = []
                            # for normal in normalInfo:
                            #     normals = filter( None, normal.replace('\n', '').split(':')[1].split(' ') )
                            #     if len(normals) != 3:
                            #         print('ERROR: Size not equal to 3\n{}'.format(normals))
                            #         exit()
                            #     if not n:
                            #         n = [float(value) for value in normals]
                            #     else:
                            #         for i in range(len(n)):
                            #             n[i] = n[i] + float(normals[i])

                            # vector = om.MVector()
                            # vector.x, vector.y, vector.z = n[0], n[1], n[2]
                            # transformed = vector * matrix
                            # if transformed.length() > 0:
                            #     transformed /= transformed.length()
                            # data.append((mesh, 'n', transformed))

                            # Store positional information
                            rotation = formatList( cmds.xform(mesh, q=1, ws=1, ro=1) )
                            data.append((mesh, 'r', rotation))
                            # translation = formatList( cmds.xform(mesh, q=1, ws=1, t=1) )
                            # data.append((mesh, 't', translation))
                            # scale = formatList( cmds.xform(mesh, q=1, ws=1, s=1) )
                            # data.append((mesh, 's', scale))

                            # Store block information
                            #data.append((mesh, 'b', int(h * xDiv + w + 1)))

                        for mesh, op, val in data:
                            key = '{}_{}'.format( op, mesh )
                            if not blockToMeshMap[h][w]:
                                blockToMeshMap[h][w].append({ key: val })
                            elif key not in blockToMeshMap[h][w][0]:
                                blockToMeshMap[h][w][0][key] = val
                            else:
                                if op == 'v':
                                    # OR with precision
                                    blockToMeshMap[h][w][0][key] = blockToMeshMap[h][w][0][key] | val
                                elif op == 'n':
                                    # Average with normal vector
                                    n = blockToMeshMap[h][w][0][key]
                                    if val.length() > 0:
                                        n = n + (val / val.length())
                                        if n.length() > 0:
                                            n /= n.length()
                                        blockToMeshMap[h][w][0][key] = n

                # Store selected faces information
                for mesh in meshes:
                    if mesh not in faceIndices:
                        continue

                    key = '{}_{}'.format( 'i', mesh )
                    faceIndices[mesh].sort()
                    faceIndexId = ','.join([str(fIndex) for fIndex in faceIndices[mesh]])
                    if faceIndexId in faceListToIdMap:
                        blockToMeshMap[h][w][0][key] = faceListToIdMap.index(faceIndexId)
                    else:
                        blockToMeshMap[h][w][0][key] = len(faceListToIdMap)
                        faceListToIdMap.append(faceIndexId)

                # Store current frame information
                # blockToMeshMap[h][w][0]['f'] = int(currTime)

                # Shift horizontally.
                left += rw * blockDim[0] / blockOffset
                right += rw * blockDim[0] / blockOffset

            # Shift vertically.
            top += rh * blockDim[1] / blockOffset
            bottom += rh * blockDim[1] / blockOffset
            left = 0
            right = rw * blockDim[0]

        # Extract test semantics
        if currTime == testTime:
            testPath = 'C:\\Users\\wesha\\Git\\deep_rendering\\python\\attngan\\data\\frame\\eval'
            make_dir(testPath)
            examplePath = 'C:\\Users\\wesha\\Git\\deep_rendering\\python\\attngan\\data\\frame\\example_filenames.txt'
            os.remove(examplePath)
            blockIndex = 1
            for i, j in testIndices:
                meshesInBlock = blockToMeshMap[i][j]
                if meshesInBlock:
                    # Screen point = Ydim * (i + 1), Xdim * (j + 1)
                    screenPoint = (i / blockOffset + 0.5) * blockDim[1], (j / blockOffset + 0.5) * blockDim[0]
                    filePath = '\\{}.txt'.format( int(blockIndex) )

                    # Store semantics for each block given the processed mesh information
                    semantics = []
                    semantics.append(extractSemantics(meshes, meshesInBlock, screenPoint, 0))
                    
                    # Write the first caption to the test block
                    blockPath = testPath + filePath
                    with open(blockPath, 'w') as f:
                        f.write( ' also '.join( semantics[0] ).replace('|', '') + '\n' )

                    # Record the index as an example caption
                    with open(examplePath, 'a') as f:
                        if blockIndex > 1:
                            f.write( '\n' )
                        f.write( 'eval/{}'.format( int(blockIndex) ) )
                    blockIndex += 1

        # Extract training semantics
        framePath = 'C:\\Users\\wesha\\Git\\deep_rendering\\python\\datasets\\Frame\\training\\{}{}\\attributes\\{:03d}'.format( dim, postfix, int(currTime) )
        make_dir(framePath)
        blockIndex = 1
        for i, data in enumerate(blockToMeshMap):
            for j, meshesInBlock in enumerate(data):
                if meshesInBlock:
                    # Screen point = Ydim * (i + 1), Xdim * (j + 1)
                    screenPoint = (i / blockOffset + 0.5) * blockDim[1], (j / blockOffset + 0.5) * blockDim[0]
                    filePath = '\\{}.txt'.format( int(blockIndex) )

                    # Store semantics for each block given the processed mesh information
                    semantics = []
                    # for frameOffset in range(5):
                    #     semantics.append(extractSemantics(meshes, meshesInBlock, screenPoint, frameOffset))
                    semantics.append(extractSemantics(meshes, meshesInBlock, screenPoint, 0))

                    # Write all captions to block in training dataset
                    blockPath = framePath + filePath
                    with open(blockPath, 'w') as f:
                        for caption in semantics:
                            f.write( ' also '.join( caption ).replace('|', '') + '\n' )
                    
                blockIndex += 1
        
        # Move to next frame
        cmds.currentTime( int(currTime) + int(stepTime), edit=True )
        currTime = int(currTime) + int(stepTime)
    
##########################
####### Run Script #######
##########################

# Display window
displayWindow()