const mysql = require('mysql2/promise');
require('dotenv').config({ path: '../.env' });


// connect to your server database, create table and insert table
async function setupDatabase() {
    const connection = await mysql.createConnection({
        host: process.env.DB_HOST,
        port: process.env.DB_PORT,
        user: process.env.DB_USERNAME,
        password: process.env.DB_PASSWORD,
        database: process.env.DB_DATABASE,
    });

    console.log('Connected to MySQL database');

    try {
        // Create departments table
        await connection.execute(`
            CREATE TABLE IF NOT EXISTS departments (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                email VARCHAR(100),
                location VARCHAR(200),
                phone_number VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        `);
        console.log('Departments table created');

        // Create employees table with foreign key to departments
        await connection.execute(`
            CREATE TABLE IF NOT EXISTS employees (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                age INT,
                position VARCHAR(100),
                username VARCHAR(50) UNIQUE,
                email VARCHAR(100),
                department_id INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (department_id) REFERENCES departments(id) ON DELETE SET NULL
            )
        `);
        console.log('Employees table created');

        // Insert fake departments
        const departments = [
            {
                name: 'Engineering',
                description: 'Responsible for software development and technical infrastructure',
                email: 'engineering@company.com',
                location: 'Building A, Floor 3',
                phone_number: '+1-555-0101'
            },
            {
                name: 'Marketing',
                description: 'Handles marketing campaigns, branding, and public relations',
                email: 'marketing@company.com',
                location: 'Building A, Floor 2',
                phone_number: '+1-555-0102'
            },
            {
                name: 'Human Resources',
                description: 'Manages employee relations, recruitment, and organizational development',
                email: 'hr@company.com',
                location: 'Building B, Floor 1',
                phone_number: '+1-555-0103'
            },
            {
                name: 'Finance',
                description: 'Oversees financial planning, accounting, and budget management',
                email: 'finance@company.com',
                location: 'Building B, Floor 2',
                phone_number: '+1-555-0104'
            },
            {
                name: 'Sales',
                description: 'Drives revenue through customer acquisition and account management',
                email: 'sales@company.com',
                location: 'Building A, Floor 4',
                phone_number: '+1-555-0105'
            }
        ];

        for (const dept of departments) {
            await connection.execute(
                'INSERT INTO departments (name, description, email, location, phone_number) VALUES (?, ?, ?, ?, ?)',
                [dept.name, dept.description, dept.email, dept.location, dept.phone_number]
            );
        }
        console.log('Inserted 5 departments');

        // Get department IDs for foreign key references
        const [deptRows] = await connection.execute('SELECT id, name FROM departments');
        const deptMap = {};
        deptRows.forEach(row => {
            deptMap[row.name] = row.id;
        });

        // Insert fake employees
        const employees = [
            { name: 'John Smith', age: 32, position: 'Senior Software Engineer', username: 'jsmith', email: 'john.smith@company.com', department: 'Engineering' },
            { name: 'Sarah Johnson', age: 28, position: 'Software Engineer', username: 'sjohnson', email: 'sarah.johnson@company.com', department: 'Engineering' },
            { name: 'Michael Chen', age: 35, position: 'Tech Lead', username: 'mchen', email: 'michael.chen@company.com', department: 'Engineering' },
            { name: 'Emily Davis', age: 26, position: 'Junior Developer', username: 'edavis', email: 'emily.davis@company.com', department: 'Engineering' },
            { name: 'David Wilson', age: 30, position: 'Marketing Manager', username: 'dwilson', email: 'david.wilson@company.com', department: 'Marketing' },
            { name: 'Jessica Brown', age: 27, position: 'Content Strategist', username: 'jbrown', email: 'jessica.brown@company.com', department: 'Marketing' },
            { name: 'Robert Taylor', age: 40, position: 'HR Director', username: 'rtaylor', email: 'robert.taylor@company.com', department: 'Human Resources' },
            { name: 'Amanda White', age: 33, position: 'HR Specialist', username: 'awhite', email: 'amanda.white@company.com', department: 'Human Resources' },
            { name: 'Christopher Lee', age: 38, position: 'Finance Manager', username: 'clee', email: 'christopher.lee@company.com', department: 'Finance' },
            { name: 'Michelle Martinez', age: 29, position: 'Accountant', username: 'mmartinez', email: 'michelle.martinez@company.com', department: 'Finance' },
            { name: 'Daniel Anderson', age: 34, position: 'Sales Director', username: 'danderson', email: 'daniel.anderson@company.com', department: 'Sales' },
            { name: 'Jennifer Thomas', age: 31, position: 'Sales Representative', username: 'jthomas', email: 'jennifer.thomas@company.com', department: 'Sales' },
            { name: 'James Garcia', age: 36, position: 'Product Manager', username: 'jgarcia', email: 'james.garcia@company.com', department: 'Engineering' },
            { name: 'Linda Rodriguez', age: 25, position: 'Marketing Coordinator', username: 'lrodriguez', email: 'linda.rodriguez@company.com', department: 'Marketing' },
            { name: 'William Clark', age: 42, position: 'Senior Accountant', username: 'wclark', email: 'william.clark@company.com', department: 'Finance' },
        ];

        for (const emp of employees) {
            const deptId = deptMap[emp.department];
            await connection.execute(
                'INSERT INTO employees (name, age, position, username, email, department_id) VALUES (?, ?, ?, ?, ?, ?)',
                [emp.name, emp.age, emp.position, emp.username, emp.email, deptId]
            );
        }
        console.log('Inserted 15 employees');

        console.log('\n=== Database setup completed successfully ===');
        
        // Display summary
        const [empCount] = await connection.execute('SELECT COUNT(*) as count FROM employees');
        const [deptCount] = await connection.execute('SELECT COUNT(*) as count FROM departments');
        console.log(`Total departments: ${deptCount[0].count}`);
        console.log(`Total employees: ${empCount[0].count}`);

    } catch (error) {
        console.error('Error setting up database:', error);
    } finally {
        await connection.end();
        console.log('Database connection closed');
    }
}

setupDatabase();
