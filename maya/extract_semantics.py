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
    meshColors = []
    for n in range(len(meshes)):
        meshColors.append([0x0, 0x0, 0x0])
    
    # Iterate over all meshes and all boundaries
    #       i % N           --> B value
    #       j % N           --> R value
    # i/(N/2) % N + j/(N/2) --> G value
    for i in range(len(blocks)):
        b = i % N
        for j in range(len(blocks[i])):
            r = j % N
            #g = (i / 4) * (N / 4) + (j / 4)
            g = (i / (N / 2)) * (N / 4) + (j / (N / 2))
            
            # Find bounds and color code for current block
            bounds = blocks[i][j]
            colorCode = [0x1 << r, 0x1 << g, 0x1 << b]
            
            # Test which meshes are contained within the block
            print('%d: Processing bounds [[%0.3f,%0.3f],[%0.3f,%0.3f]]' % (g, bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))
                    
    for k, mesh in enumerate(meshes):
        updateShaderColor(mesh, meshColors[k], 2**N)
        print(mesh, meshColors[k])
    
##########################
####### Run Script #######
##########################

# Display window
displayWindow()