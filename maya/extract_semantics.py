import maya.cmds as cmds
import maya.OpenMaya as om
import maya.OpenMayaUI as omui
from functools import partial
import json as json
import os as os

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
    
# Return the bit code for shader inputs and block offsets
def bitCode(mesh, r, g, b):
    shader = findShader(mesh)
    rVal = cmds.getAttr ( (shader) + '.r' )
    gVal = cmds.getAttr ( (shader) + '.g' )
    bVal = cmds.getAttr ( (shader) + '.b' ) 
    
    return [(rVal >> r) & 0x1, (gVal >> g) & 0x1, (bVal >> b) & 0x1]

# Test if the color value implies block intersection
def checkBitCode(code):
    if code[0] == 1 and code[1] == 1 and code[2] == 1:
        return True
        
    return False

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
def extractSemantics(meshes, screenPoint, neighbors, cutoff):
    semantics = []
    
    for mesh in meshes:
        semanticsForMesh = []
        semanticsForNeighbors = []
        
        # Calculate semantics for mesh
        translation = formatList( cmds.xform(mesh, q=1, ws=1, t=1) )
        rotation = formatList( cmds.xform(mesh, q=1, ws=1, rp=1) )
        scaling = formatList( cmds.xform(mesh, q=1, ws=1, s=1) )
        worldPoint = screenSpaceToWorldSpace(screenPoint)
        d = postionDistance(meshPosition(mesh), worldPoint)
        print(translation)
        semanticsForMesh.append('d : {0:.6f}'.format( d ))
        semanticsForMesh.append('t : [{0:.6f}, {0:.6f}, {0:.6f}]'.format( translation[0], translation[1], translation[2] ))
        semanticsForMesh.append('r : [{0:.6f}, {0:.6f}, {0:.6f}]'.format( rotation[0], rotation[1], rotation[2] ))
        semanticsForMesh.append('s : [{0:.6f}, {0:.6f}, {0:.6f}]'.format( scaling[0], scaling[1], scaling[2] ))
        
        semantics.append('{} : {}'.format( mesh, semanticsForMesh ))
                
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

##########################
### Main Functionality ###
##########################

# Create and display menu system
def displayWindow():
    menu = cmds.window( title="Extract Semantics Tool", iconName='ExtractSemanticsTool', widthHeight=(350, 400) )
    scrollLayout = cmds.scrollLayout( verticalScrollBarThickness=16 )
    cmds.flowLayout( columnSpacing=10 )
    cmds.columnLayout( cat=('both', 25), rs=10, cw=340 )
    cmds.text( label="\nThis is the \"Extract Sematics Tool\"! This tool will extract semantics for the loaded scene.\n\n", ww=True, al="left" )
    cmds.text( label="To run:\n1) Input the information in the fields below.\n2) Click \"Run\".", al="left" )
    cmds.text( label='Enter the keyframe at which to start semantics generation (1):', al='left', ww=True )
    startTimeField = cmds.textField()
    cmds.text( label='Enter the keyframe at which to end semantics generation (1):', al='left', ww=True )
    endTimeField = cmds.textField()
    cmds.text( label='Enter the step at which to process frames (1):', al='left', ww=True )
    stepTimeField = cmds.textField()
    cmds.text( label='Enter the cut off distance for per-object semantics (100):', al='left', ww=True )
    cutOffField = cmds.textField()
    cmds.button( label='Run', command=partial( generateSemantics, menu, startTimeField, endTimeField, stepTimeField, cutOffField ) )
    cmds.text( label="\n", al='left' )
    cmds.showWindow( menu )

def generateSemantics( menu, startTimeField, endTimeField, stepTimeField, cutOffField, *args ):
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
    cutOff = cmds.textField(cutOffField, q=True, tx=True )
    if (cutOff == ''):
        print 'WARNING: Default cutoff (100) used...'
        cutOff = '100'
    cmds.deleteUI( menu, window=True )
    
    # Set up program
    resWidth = cmds.getAttr('defaultResolution.width')
    resHeight = cmds.getAttr('defaultResolution.height')
    blockDim = 0 # Placeholder
                
    # Obtain all meshes in the scene
    currSel = cmds.ls()
    meshes = collectObjects(currSel)
    meshBlocks = {}
    
    # Iterate over all meshes
    xNum, yNum = None, None
    blocks = []
    blockToMeshMap = []
    for k, mesh in enumerate(meshes):
        shader = findShader(mesh)
        N = cmds.getAttr ( (shader) + '.n' )
        
        if blockDim == 0:
            blockDim = [int(resWidth / (2 * N)), int(resHeight / ((N / 8) * N))]
        
        xDiv = float(resWidth) / blockDim[0]
        yDiv = float(resHeight) / blockDim[1]
        step = (resWidth / blockDim[0]) / (N / 2)
        
        # Set up blocks
        if xNum is None or yNum is None:
            for h in range(int(yDiv)):
                row = []
                blockToMeshMap.append([])
                
                # Find boundaries for each block in the row
                top = h / yDiv
                bottom = (h + 1) / yDiv
                for w in range(int(xDiv)):
                    left = w / xDiv
                    right = (w + 1) / xDiv
                    
                    row.append([[left,right],[top,bottom]])
                    blockToMeshMap[h].append([])
                    
                # Append the finished row
                blocks.append(row)
            
            yNum = len(blocks)
            xNum = len(blocks[0])
            print('Block Dim: (%d, %d), Blocks: (%d, %d)' % (blockDim[0], blockDim[1], len(blocks[0]), len(blocks)))
    
        # Iterate over all boundaries
        print('Evaluating {}...'.format( mesh ))
        for i in range(yNum):
            b = i % N
            for j in range(xNum):
                r = j % N
                g = int((i / (N / 2))) * int(step) + int((j / (N / 2)))
                            
                # Check bit code of mesh for current block
                code = bitCode(mesh, r, g, b)
                if checkBitCode(code):
                    if mesh in meshBlocks:
                        meshBlocks[mesh].append([ r, g, b ])
                        blockToMeshMap[i][j].append(mesh)
                    else:
                        meshBlocks[mesh] = [[ r, g, b ]]
                        blockToMeshMap[i][j] = [mesh]
                        
    # Check if the algorithm correctly extracted the blocks
    for k, mesh in enumerate(meshes):
        meshColors = [0x0, 0x0, 0x0]
        if mesh in meshBlocks:
            for c in meshBlocks[mesh]:
                colorCode = [0x1 << c[0], 0x1 << c[1], 0x1 << c[2]]
                for n in range(len(colorCode)):
                    meshColors[n] |= colorCode[n]
        else:
            print('{}: No blocks found!'.format( mesh ))
        
        shader = findShader(mesh)
        rVal = cmds.getAttr ( (shader) + '.r' )
        gVal = cmds.getAttr ( (shader) + '.g' )
        bVal = cmds.getAttr ( (shader) + '.b' )
        if (meshColors[0] == rVal) and (meshColors[1] == gVal) and (meshColors[2] == bVal):
            print('{}: Good!'.format( mesh ))
        else:
            print('{}: {} ({},{},{})'.format( mesh, meshColors, rVal, gVal, bVal ))
            
    # Extract semantics for each mesh
    semantics = []
    for i, data in enumerate(blockToMeshMap):
        row = []
        for j, meshesInBlock in enumerate(data):
            if not meshesInBlock:
                print('No semantics for block({},{})'.format( i, j ))
            else:
                # Screen point = Ydim * (i + 1), Xdim * (j + 1)
                screenPoint = blockDim[1] * (i + 0.5), blockDim[0] * (j + 0.5)
                row.append('({}, {}) : {}'.format( i, j, extractSemantics(meshesInBlock, screenPoint, meshes, float(cutOff)) ))
        semantics.append(row)
    
    for row in semantics:
        print(row)
    
    # Write data to an output file
    filepath = cmds.file(q=True, sn=True)
    filename = os.path.basename(filepath)
    raw_name, extension = os.path.splitext(filename)
    with open('C:\\Users\\wesha\\Documents\\maya\\projects\\CS5800\\scenes\\{}_output_{}.txt'.format( raw_name, N ), 'w') as f:
        f.write( json.dumps(semantics).replace('"', '').replace('\'', '').replace('\\', '') )
    
##########################
####### Run Script #######
##########################

# Display window
displayWindow()