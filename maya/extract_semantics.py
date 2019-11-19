import maya.cmds as cmds
import maya.OpenMaya as om

##########################
#### Helper Functions ####
##########################

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
    n = cmds.getAttr ( (shader) + '.n' )
    
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

##########################
### Main Functionality ###
##########################

# Create and display menu system
def displayWindow():
    menu = cmds.window( title="Extract Semantics Tool", iconName='SemanticsTool', widthHeight=(350, 400) )
    scrollLayout = cmds.scrollLayout( verticalScrollBarThickness=16 )
    cmds.flowLayout( columnSpacing=10 )
    cmds.columnLayout( cat=('both', 25), rs=10, cw=340 )
    cmds.text( label="\nThis is the \"Extract Sematics Tool\"! This tool will extract semantics for the loaded scene.\n\n", ww=True, al="left" )
    cmds.text( label="To run:\n1) Input the information in the fields below.\n2) Click \"Run\".", al="left" )
    cmds.text( label='Enter the keyframe at which to start semantics generation (1):', al='left', ww=True )
    startTimeField = cmds.textField()
    cmds.text( label='Enter the keyframe at which to end semantics generation (120):', al='left', ww=True )
    endTimeField = cmds.textField()
    cmds.text( label='Enter the step at which to process frames (1):', al='left', ww=True )
    stepTimeField = cmds.textField()
    cmds.text( label='Enter the number of bits used to store each, a multiple of 8 is recommended (8):', al='left', ww=True )
    bitNumField = cmds.textField()
    cmds.button( label='Run', command=partial( generateSemantics, menu, startTimeField, endTimeField, stepTimeField, bitNumField ) )
    cmds.text( label="\n", al='left' )
    cmds.showWindow( menu )

def generateSemantics( menu, startTimeField, endTimeField, stepTimeField, bitNumField, *args ):
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
    
    # Set up blocks
    blocks = []
    for w in range(resWidth / blockDim[0]):
        row = []
        
        # Find boundaries for each block in the row
        left = (w * blockDim[0]) / float(resWidth)
        right = ((w + 1) * blockDim[0]) / float(resWidth)
        for h in range(resHeight / blockDim[1]):
            top = (h * blockDim[1]) / float(resHeight)
            bottom = ((h + 1) * blockDim[1]) / float(resHeight)
            
            row.append([[left,right],[top,bottom]])
            
        # Append the finished row
        blocks.append(row)
            
    print('Block Dim: (%d, %d), Blocks: (%d, %d)' % (blockDim[0], blockDim[1], len(blocks), len(blocks[0])))
    
    # Obtain all meshes in the scene
    currSel = cmds.ls()
    meshes = collectObjects(currSel)
    meshBlocks = {}
    
    # Iterate over all meshes and all boundaries
    step = (resWidth / blockDim[0]) / (N / 2)
    for i in range(len(blocks)):
        b = i % N
        for j in range(len(blocks[i])):
            r = j % N
            g = int((j / (N / 2))) * int(step) + int((i / (N / 2)))
            
            # Check bit code for current block
            for k, mesh in enumerate(meshes):
                code = bitCode(mesh, r, g, b)
                if checkBitCode(code):
                    if mesh in meshBlocks:
                        meshBlocks[mesh].append([r,g,b])
                    else:
                        meshBlocks[mesh] = [[r,g,b]]
    
    for k, mesh in enumerate(meshes):
        for blockOffset in meshBlocks[mesh]:
            print('%s: %d, %d, %d' % (mesh, blockOffset[0], blockOffset[1], blockOffset[2]))
    
##########################
####### Run Script #######
##########################

# Display window
displayWindow()