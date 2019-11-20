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
    menu = cmds.window( title="Extract Semantics Tool", iconName='ExtractSemanticsTool', widthHeight=(350, 400) )
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
    cmds.button( label='Run', command=partial( generateSemantics, menu, startTimeField, endTimeField, stepTimeField ) )
    cmds.text( label="\n", al='left' )
    cmds.showWindow( menu )

def generateSemantics( menu, startTimeField, endTimeField, stepTimeField, *args ):
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
    cmds.deleteUI( menu, window=True )
    
    # Set up program
    resWidth = cmds.getAttr('defaultResolution.width')
    resHeight = cmds.getAttr('defaultResolution.height')
                
    # Obtain all meshes in the scene
    currSel = cmds.ls()
    meshes = collectObjects(currSel)
    meshBlocks = {}
    
    # Iterate over all meshes
    xNum, yNum = None, None
    blocks = []
    for k, mesh in enumerate(meshes):
        shader = findShader(mesh)
        N = cmds.getAttr ( (shader) + '.n' )
        blockDim = [int(resWidth / (2 * N)), int(resHeight / ((N / 8) * N))]
        xDiv = float(resWidth) / blockDim[0]
        yDiv = float(resHeight) / blockDim[1]
        step = (resWidth / blockDim[0]) / (N / 2)
        
        # Set up blocks
        if xNum is None or yNum is None:
            for h in range(int(yDiv)):
                row = []
                
                # Find boundaries for each block in the row
                left = h / yDiv
                right = (h + 1) / yDiv
                for w in range(int(xDiv)):
                    top = w / xDiv
                    bottom = (w + 1) / xDiv
                    
                    row.append([[left,right],[top,bottom]])
                    
                # Append the finished row
                blocks.append(row)
            
            yNum = len(blocks)
            xNum = len(blocks[0])
    
        # Iterate over all boundaries
        for i in range(yNum):
            b = i % N
            for j in range(xNum):
                r = j % N
                g = int((i / (N / 2))) * int(step) + int((j / (N / 2)))
                            
                # Check bit code for current block
                for k, mesh in enumerate(meshes):
                    code = bitCode(mesh, r, g, b)
                    if checkBitCode(code):
                        if mesh in meshBlocks:
                            meshBlocks[mesh].append([r,g,b])
                        else:
                            meshBlocks[mesh] = [[r,g,b]]
                        
    # Check if the algorithm correctly extracted the blocks.
    for k, mesh in enumerate(meshes):
        meshColors = [0x0, 0x0, 0x0]
        if mesh in meshBlocks:
            for c in meshBlocks[mesh]:
                colorCode = [0x1 << c[0], 0x1 << c[1], 0x1 << c[2]]
                for n in range(len(colorCode)):
                    meshColors[n] |= colorCode[n]
        
        shader = findShader(mesh)
        rVal = cmds.getAttr ( (shader) + '.r' )
        gVal = cmds.getAttr ( (shader) + '.g' )
        bVal = cmds.getAttr ( (shader) + '.b' )
        if (meshColors[0] == rVal) and (meshColors[1] == gVal) and (meshColors[2] == bVal):
            print(mesh, 'Good!')
        else:
            print(mesh, meshColors, rVal, gVal, bVal)        
    
##########################
####### Run Script #######
##########################

# Display window
displayWindow()